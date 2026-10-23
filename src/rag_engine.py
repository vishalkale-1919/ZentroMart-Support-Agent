"""RAG Engine: Document chunking, indexing, precedence weighting, and hybrid retrieval."""
import re
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from src.config import KB_DIR, DATA_DIR, REFERENCE_DATE

# Document Precedence weights based on KB-00.1
PRECEDENCE_WEIGHTS = {
    "policy": 1.4,
    "agent_guide": 1.25,
    "help": 1.15,
    "faq": 1.05,
    "catalog": 1.1,
    "meta": 1.0,
    "marketing": 0.85
}

class DocumentChunk:
    def __init__(self,
                 chunk_id: str,
                 section_id: str,
                 doc_id: str,
                 doc_title: str,
                 doc_type: str,
                 status: str,
                 effective_from: str,
                 title: str,
                 content: str,
                 source_file: str):
        self.chunk_id = chunk_id
        self.section_id = section_id
        self.doc_id = doc_id
        self.doc_title = doc_title
        self.doc_type = doc_type
        self.status = status
        self.effective_from = effective_from
        self.title = title
        self.content = content
        self.source_file = source_file
        
        base_weight = PRECEDENCE_WEIGHTS.get(doc_type.lower(), 1.0)
        if status.lower() == "superseded":
            base_weight *= 0.35  # Penalty unless explicitly requested
        self.precedence_weight = base_weight

    def full_search_text(self) -> str:
        return f"{self.section_id} {self.title} {self.doc_title} {self.content}"

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "section_id": self.section_id,
            "doc_id": self.doc_id,
            "doc_title": self.doc_title,
            "doc_type": self.doc_type,
            "status": self.status,
            "effective_from": self.effective_from,
            "title": self.title,
            "content": self.content,
            "source_file": self.source_file,
            "precedence_weight": self.precedence_weight
        }

class RAGEngine:
    def __init__(self):
        self.chunks: List[DocumentChunk] = []
        self.chunk_by_section_id: Dict[str, DocumentChunk] = {}
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.products_df: Optional[pd.DataFrame] = None
        self.product_lookup: Dict[str, dict] = {}
        self._load_and_index()

    def _parse_markdown_doc(self, file_path: Path) -> List[DocumentChunk]:
        text = file_path.read_text(encoding="utf-8")
        
        meta = {
            "doc_id": "",
            "title": "",
            "doc_type": "help",
            "effective_from": "2026-01-01",
            "status": "current"
        }
        
        frontmatter_match = re.match(r'^---\s*\n(.*?)\n---\s*\n', text, re.DOTALL)
        content_text = text
        if frontmatter_match:
            raw_meta = frontmatter_match.group(1)
            content_text = text[frontmatter_match.end():]
            for line in raw_meta.splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip().lower()] = v.strip().strip('"\'')

        sections = re.split(r'(?=^##\s*\[[^\]]+\])', content_text, flags=re.M)
        chunks = []
        for sec in sections:
            sec = sec.strip()
            if not sec:
                continue
            m = re.match(r'^##\s*\[([^\]]+)\]\s*(.*)', sec)
            if m:
                sec_id = m.group(1).strip()
                sec_title = m.group(2).strip()
                body = sec[m.end():].strip()
            else:
                sec_id = meta["doc_id"] or file_path.stem
                sec_title = meta["title"] or file_path.stem
                body = sec

            chunk = DocumentChunk(
                chunk_id=f"{file_path.stem}::{sec_id}",
                section_id=sec_id,
                doc_id=meta.get("doc_id", ""),
                doc_title=meta.get("title", ""),
                doc_type=meta.get("doc_type", "help"),
                status=meta.get("status", "current"),
                effective_from=meta.get("effective_from", "2026-01-01"),
                title=sec_title,
                content=body,
                source_file=file_path.name
            )
            chunks.append(chunk)

        return chunks

    def _load_and_index(self):
        self.chunks = []
        self.chunk_by_section_id = {}
        self.product_lookup = {}

        # 1. Index 21 Markdown files from kb/
        if KB_DIR.exists():
            for f in sorted(KB_DIR.glob("*.md")):
                doc_chunks = self._parse_markdown_doc(f)
                for c in doc_chunks:
                    self.chunks.append(c)
                    self.chunk_by_section_id[c.section_id] = c

        # 2. Index products from data/products.csv as products.csv:{sku}
        products_file = DATA_DIR / "products.csv"
        if products_file.exists():
            try:
                self.products_df = pd.read_csv(products_file, dtype=str)
                for _, row in self.products_df.iterrows():
                    sku = str(row["sku"]).strip()
                    title = row["title"]
                    self.product_lookup[sku.upper()] = row.to_dict()
                    self.product_lookup[title.lower()] = row.to_dict()
                    
                    sec_id = f"products.csv:{sku}"
                    chunk_title = f"Product {sku}: {title}"
                    body = (
                        f"SKU: {sku}. Product Title: {title}. Brand: {row['brand']}. "
                        f"Category: {row.get('category', '')}. Return Category: {row['return_category']}. "
                        f"Price: INR {row['price_inr']}. Fulfilled By: {row['fulfilled_by']}. "
                        f"Assured: {row['assured']}. Warranty: {row['warranty_months']} months. "
                        f"Description: {row['description']}"
                    )
                    chunk = DocumentChunk(
                        chunk_id=f"products::{sku}",
                        section_id=sec_id,
                        doc_id="CATALOG",
                        doc_title="ZentroMart Product Catalog",
                        doc_type="catalog",
                        status="current",
                        effective_from="2026-01-01",
                        title=chunk_title,
                        content=body,
                        source_file="products.csv"
                    )
                    self.chunks.append(chunk)
                    self.chunk_by_section_id[sec_id] = chunk
            except Exception as e:
                print(f"Warning: Failed to index products.csv: {e}")

        # 3. Build TF-IDF search index
        corpus = [c.full_search_text() for c in self.chunks]
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            token_pattern=r'(?u)\b[\w\-\.\:]+\b',
            sublinear_tf=True
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.08) -> List[Tuple[DocumentChunk, float]]:
        """Retrieve most relevant chunks with hybrid scoring, precedence, and concept boosts."""
        if not query or not query.strip() or self.tfidf_matrix is None:
            return []

        q_lower = query.lower()

        # Check for explicit SKU references in query
        sku_matches = re.findall(r'\b([A-Z]{3}-\d{4})\b', query.upper())
        explicit_product_chunks = []
        matched_categories = set()

        for sku in sku_matches:
            sec_key = f"products.csv:{sku}"
            if sec_key in self.chunk_by_section_id:
                explicit_product_chunks.append(self.chunk_by_section_id[sec_key])
            if sku in self.product_lookup:
                matched_categories.add(self.product_lookup[sku].get('return_category', ''))

        # Also match product title keywords from query if no SKU was provided
        if not sku_matches and self.products_df is not None:
            for _, row in self.products_df.iterrows():
                t_lower = row['title'].lower()
                words = [w for w in t_lower.split() if len(w) > 3 and w not in ('black', 'white', 'grey', 'beige', 'blue', 'green', 'with', 'inch')]
                if len(words) >= 2 and all(w in q_lower for w in words[:2]):
                    sku = row['sku']
                    sec_key = f"products.csv:{sku}"
                    if sec_key in self.chunk_by_section_id and self.chunk_by_section_id[sec_key] not in explicit_product_chunks:
                        explicit_product_chunks.append(self.chunk_by_section_id[sec_key])
                        matched_categories.add(row.get('return_category', ''))

        # Intent & Concept Analysis
        asking_for_superseded = bool(
            re.search(r'before\s*(?:1|01)?\s*aug|superseded|purana|purani\s*policy|v3|july\s*ko\s*deliver|delivered\s*in\s*july|delivered\s*before\s*august', q_lower)
        )
        is_opened_beauty = bool(
            re.search(r'(?:opened|seal|खोल|open).*?(?:lipstick|perfume|parfum|beauty|creams?|lotion)', q_lower) or
            re.search(r'(?:lipstick|perfume|parfum|beauty).*?(?:opened|seal|खोल|open)', q_lower)
        )
        is_return_window_general = bool(
            re.search(r'kitne\s*din|कितने\s*दिन|return\s*window|how\s*many\s*days.*?return', q_lower)
        )
        is_unauthorized_order = bool(
            re.search(r'(?:order|account|payment).*?(?:never\s*(?:made|placed)|didn\'t\s*(?:make|place)|did\s*not\s*(?:make|place)|fraud|unauthorized)', q_lower) or
            re.search(r'maine\s*(?:order|payment)\s*nahi\s*(?:kiya|place)', q_lower)
        )
        is_shipped_cancellation = bool(
            re.search(r'(?:already\s*shipped|on\s*its\s*way|shipped).*?(?:cancel|stop|रोक|कैंसिल)', q_lower) or
            re.search(r'(?:cancel|stop|रोक|कैंसिल).*?(?:already\s*shipped|on\s*its\s*way|shipped)', q_lower)
        )
        is_cod_refund = bool(
            re.search(r'(?:cash\s*on\s*delivery|cod).*?(?:refund|money|paisa|रिफ़ंड|वापस)', q_lower) or
            re.search(r'कैश\s*ऑन\s*डिलीवरी.*?रिफ़ंड', q_lower)
        )
        is_collection_damaged_fee = bool(
            re.search(r'(?:collection\s*free|pickup\s*fee|reverse\s*pickup).*?(?:damaged|defective|wrong|fee|torn|came\s*torn)', q_lower) or
            re.search(r'(?:damaged|defective|wrong|torn|came\s*torn).*?(?:collection\s*free|pickup\s*fee|reverse\s*pickup)', q_lower)
        )
        is_electronics_10day_change = bool(
            re.search(r'(?:still\s*ten\s*days|10\s*days).*?electronics', q_lower) or
            re.search(r'electronics.*?(?:still\s*ten\s*days|10\s*days|7\s*days)', q_lower)
        )

        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        scored_chunks = []
        for idx, chunk in enumerate(self.chunks):
            raw_sim = sims[idx]
            weight = chunk.precedence_weight
            
            if asking_for_superseded and "superseded" in chunk.status:
                weight = 1.6
            elif not asking_for_superseded and "superseded" in chunk.status:
                weight = 0.2

            final_score = raw_sim * weight

            # Bonus for explicit product match
            if chunk.section_id.startswith("products.csv:") and any(s in chunk.section_id for s in sku_matches):
                final_score += 0.45

            # Concept boosts
            if is_opened_beauty and chunk.section_id in ("RET-4", "RET-2"):
                final_score += 0.45

            if is_return_window_general and chunk.section_id == "RET-2":
                final_score += 0.40

            if is_unauthorized_order and chunk.section_id == "ESC-1":
                final_score += 0.55

            if is_shipped_cancellation and chunk.section_id == "CAN-2":
                final_score += 0.50

            if is_cod_refund and chunk.section_id in ("REF-3", "REF-2"):
                final_score += 0.45

            if is_collection_damaged_fee and chunk.section_id in ("RET-6", "DWM-2"):
                final_score += 0.45

            if is_electronics_10day_change and chunk.section_id in ("RET-3", "RET-2"):
                final_score += 0.45

            if asking_for_superseded and chunk.section_id in ("RET-3", "RET-2", "RETv3-2"):
                final_score += 0.35

            if "assured" in q_lower and chunk.section_id == "WAR-2":
                final_score += 0.35

            if ("refund" in q_lower or "रिफ़ंड" in q_lower or "wallet" in q_lower) and chunk.section_id == "REF-2":
                final_score += 0.25

            if ("plus" in q_lower or "सदस्य" in q_lower) and chunk.section_id in ("PLS-2", "RET-6", "MKT-1"):
                final_score += 0.25

            if matched_categories and chunk.section_id == "RET-2":
                final_score += 0.15

            if final_score >= score_threshold:
                scored_chunks.append((chunk, final_score))

        # Add explicit product chunks if matched
        for p_chunk in explicit_product_chunks:
            if not any(c.section_id == p_chunk.section_id for c, _ in scored_chunks):
                scored_chunks.append((p_chunk, 0.95))

        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return scored_chunks[:top_k]

    def get_chunk_by_id(self, section_id: str) -> Optional[DocumentChunk]:
        return self.chunk_by_section_id.get(section_id.strip())

# Global singleton RAG engine
rag_engine = RAGEngine()
