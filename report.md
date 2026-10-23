# VoiceOps 2026: Capstone Project Report
**LLM & Generative AI Course · Consulting & Analytics Club, IIT Guwahati × Gnani.ai**  
**Project:** End-to-End Multimodal AI Customer Support Agent for ZentroMart  
**Reference Date:** 20 October 2026  

---

## 1. Executive Summary

This report documents the architectural design, implementation, and empirical evaluation of **VoiceOps 2026**, an end-to-end multimodal customer support agent for **ZentroMart**, a fictional Indian online store selling groceries and electronics.

The agent guides customers across text and spoken channels, operating in four sequential stages:
1. **Step 1 (Prompting LLM):** System prompting establishing AI assistant identity and behavioral boundaries.
2. **Step 2 (RAG Grounding):** Document chunking, precedence weighting, Section ID tracking, and strict hallucination deterrence.
3. **Step 3 (Conversational Intelligence & Verification):** Multi-turn context windows, strict two-step identity verification gating, mock tool integration (`lookup_order`, `create_ticket`), and escalation policies.
4. **Step 4 (VoiceOps Integration):** Speech-to-Text (STT) and Text-to-Speech (TTS) integration using **Gnani.ai** models (`timbre-v2.5`), text normalization for speech, and batch execution capabilities.

---

## 2. Architecture & Step-by-Step Implementation

### Step 1: Prompting Baseline & Assistant Persona
In Step 1, the agent was configured with a comprehensive system prompt establishing its role as ZentroMart's AI support assistant. As mandated by `agent_conversation_style.md` [STY-1], the agent's very first reply clearly discloses its AI identity in both English and Indian languages (e.g., "I am ZentroMart's AI support assistant" / "मैं ZentroMart का AI सहायता सहायक हूँ"). 

In baseline mode (prompting only), the LLM lacks access to internal policy documents and catalog records, relying purely on parametric memory. This established a quantitative baseline for measuring hallucination rates and factual grounding.

### Step 2: Knowledge Base Indexing & Hybrid RAG Engine
To eliminate hallucinations and ground answers in company truth, we constructed a high-precision RAG engine indexing all 21 company documents in `kb/` and 70 products in `data/products.csv`:
- **Granular Chunking:** Markdown documents were segmented by section headers (`## [SECTION-ID] Section Title`). Each chunk retained its metadata (`doc_id`, `section_id`, `doc_type`, `status`, `effective_from`).
- **Product Catalog Indexing:** Every catalog item was indexed under the standardized key `products.csv:{sku}` with brand, category, price, and ZentroMart Assured status.
- **Precedence Hierarchy Enforcement (`KB-00.1`):** A strict multiplier was implemented prioritizing:
  $$\text{Current Policy (1.4)} > \text{Agent Guides (1.25)} > \text{Help (1.15)} > \text{FAQ (1.05)} > \text{Marketing (0.85)}$$
  Superseded policies (such as `pol_returns_v3_superseded.md`) receive a heavy penalty (0.35) unless the query explicitly concerns orders delivered before 1 August 2026.
- **Strict Out-of-Scope Handling:** When retrieved confidence falls below threshold or queries target out-of-scope policies (e.g., midnight delivery, GST invoices for bulk orders, physical stores in Kochi), the agent gracefully states that the policy does not contain this information and offers human assistance.

### Step 3: Multi-Turn Conversation, Identity Verification & Tool Integration
The conversational engine implements a state machine maintaining dialogue state across multiple customer turns:
- **Identity Verification Gating (`agent_identity_verification.md`):**
  - General policy queries require no caller verification.
  - For order-specific lookups, the agent invokes `lookup_order(order_id)`.
  - Details remain strictly withheld until the caller supplies ONE valid proof: the **last 4 digits of the registered phone** OR the **delivery pincode**.
  - A maximum of **one retry** is permitted upon mismatch. If verification fails twice, order disclosure is refused.
- **Date & Calendar Rules:**
  - Business days run **Monday to Saturday**, skipping the 4 warehouse holidays in `data/holidays.csv` (15 Aug, 02 Oct, 19 Oct, 09 Nov 2026).
  - Return windows use calendar days from delivery date (e.g., 7 days for electronics delivered after 1 Aug 2026; 30 days for Festive Sale fashion orders placed between 1–15 Oct 2026).
- **Human Escalation Protocol (`agent_escalation_handoff.md`):**
  - Triggers: Claims for damaged items $> ₹10,000$ (Priority Resolution), suspected fraud, repeat contacts ($\ge 2$ prior calls), safety hazards, and overdue refunds past max business-day timelines + grace period.
  - Automatically invokes `create_ticket(summary=...)` with structured summaries $\ge 20$ characters before confirming handover.

### Step 4: Voice Integration via Gnani.ai Models
The voice layer bridges speech and conversational intelligence:
- **Gnani STT:** Transcribes customer audio (supports `en-IN`, `hi-IN`, `hi-en`, and regional languages) with recovery from silence/unclear audio.
- **Ear Text Normalization:** Formats text for natural speech: spaces out 8-digit order numbers (`8 9 7 8, 2 8 3 7`), spells out rupee currency terms (`49 rupees`), and expresses dates with month names (`the 24th of October 2026`).
- **Gnani TTS:** Synthesizes voice audio using model `timbre-v2.5` and voice `Kaveri`.
- **Batch Evaluation Runner (`run.py`):** Accepts an input directory of customer audio recordings, generates transcripts, executes conversational logic, retrieves chunks, and saves replies (`.txt`, `.wav`, `_chunks.json`, `batch_summary.json`).

---

## 3. Evaluation Results: Step 1 vs. Step 2 vs. Step 3

The system was evaluated against the 52 questions in `practice/questions.csv` and the 23 conversations (96 customer turns) in `practice/conversations.csv`:

### Step 1 vs. Step 2 Performance Comparison

| Evaluation Metric | Step 1 (Prompting Baseline) | Step 2 (RAG Grounded) | Delta / Impact |
| :--- | :---: | :---: | :---: |
| **Turn Decision Accuracy** | 88.46% | **88.46%** | Grounded consistency |
| **Gold Evidence Section Recall** | 0.00% | **97.87%** | $+97.87\%$ (Accurate retrieval) |
| **Hallucination on Unanswerable Queries** | 60.00% | **0.00%** | **$-60.00\%$ (Hallucinations eliminated)** |
| **Unanswerable Policy Detection** | 0.00% | **100.00%** | $+100.00\%$ (Polite boundary) |
| **Must-Not-Include Violations** | 4 violations | **0 violations** | Zero policy leaks |

### Step 3: Multi-Turn Conversation & Tool Execution (96 Turns)

| Multi-Turn Metric | Evaluated Score | Benchmark Requirement |
| :--- | :---: | :---: |
| **Conversation Decision Accuracy** | **100.00% (96/96)** | Exceeds expected benchmark |
| **Tool Call Invocation Accuracy** | **100.00% (96/96)** | Accurate `lookup_order` & `create_ticket` |
| **Verification Gate Enforcement** | **100.00%** | Zero private data leakage before proof |
| **Must-Include Dialogue Score** | **92.71% (89/96)** | High conversational adherence |
| **Escalation Protocol Compliance** | **100.00%** | Valid summaries generated |

### Key Learnings
1. **Parametric Hallucination vs Grounded Truth:** Without RAG, LLMs hallucinate non-existent student discounts and store pickup options. Grounded retrieval with explicit unanswerability thresholds completely eradicated false claims.
2. **Precedence Resolution:** Real customer policies conflict (e.g. general return FAQs stating 10 days vs current v4 policy establishing 7 days for electronics). Applying precedence weights ensured current policies consistently superseded older guides.
3. **Identity-Checked Safety:** Stateless tools require stateful conversation managers. The session state machine successfully tracked verification attempts, preventing brute-force disclosures while permitting legitimate retries.

---

## 4. Challenges & Engineering Solutions

1. **ASR Dialect & Multilingual Variances:**
   - *Challenge:* Practice questions contained phonetic ASR variations (e.g., "ele-zero zero one to", "aakhri char ank", "kitne din me wapas").
   - *Solution:* Implemented dual lexical-semantic matching with regex token normalization mapping Hindi/Hinglish colloquialisms to canonical catalog and policy concepts.
2. **Business Day & Holiday Arithmetic:**
   - *Challenge:* Calculating whether a refund is overdue required navigating Monday–Saturday business days while skipping specific warehouse closure holidays (e.g., Diwali / festival holidays).
   - *Solution:* Built a calendar utility integrating `data/holidays.csv` that precisely computes dates (e.g., mapping an initiation on Saturday 17 October to 27 October 2026).
3. **Voice Ear Normalization:**
   - *Challenge:* Raw LLM markdown output (asterisks, bullet points, raw 8-digit numbers) sounds robotic and confusing over TTS.
   - *Solution:* Developed a speech normalizer converting `₹` to `rupees`, expanding dates into spoken prose, and inserting space pauses into order IDs.

---

## 5. Bonus Deliverables & Enhancements

- **Interactive Streamlit Web Interface (`app.py`):** Live voice recording, audio playback, order lookup simulator, escalation ticket log, and real-time telemetry inspector.
- **Dual Cloud & Offline Resilience:** Fully integrated with Google Gemini and Groq API keys, with automatic deterministic fallback ensuring reproducible execution in offline evaluation environments.
- **Standardized Submission Structure:** Complete with `run.py`, `run.sh`, `requirements.txt`, `.env.example`, and comprehensive documentation.
