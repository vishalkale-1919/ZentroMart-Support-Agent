"""VoiceOps 2026: ZentroMart Voice & Chat Support Agent - Streamlit Application."""
import os
import sys
import tempfile
import time
from pathlib import Path
import streamlit as st
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    REFERENCE_DATE_STR, GNANI_API_KEY, GEMINI_API_KEY, GROQ_API_KEY,
    LLM_PROVIDER, GNANI_TTS_VOICE, GNANI_TTS_MODEL, OUTPUT_DIR
)
from src.conversation_agent import ConversationAgent
from src.voice_service import voice_service
from src.rag_engine import rag_engine
from src.tools_adapter import lookup_order, create_ticket
from src.data_loader import data_loader
from src.utils import format_for_speech

# Page config
st.set_page_config(
    page_title="ZentroMart VoiceOps 2026",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
if "agent" not in st.session_state:
    st.session_state.agent = ConversationAgent(mode="rag")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "session_id" not in st.session_state:
    st.session_state.session_id = f"session_{int(time.time())}"

# Sidebar
with st.sidebar:
    st.title("🎙️ VoiceOps 2026")
    st.caption("IIT Guwahati × Gnani.ai Capstone Project")
    st.markdown("---")
    
    st.subheader("System Status")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Ref Date", REFERENCE_DATE_STR)
    with col2:
        st.metric("KB Docs", "21 MDs")

    st.markdown("---")
    st.subheader("API Keys & Providers")
    st.write(f"**Gnani Speech API:** {'🟢 Configured' if GNANI_API_KEY else '🟡 Mock/Fallback'}")
    st.write(f"**Gemini API:** {'🟢 Configured' if GEMINI_API_KEY else '🟡 Offline Engine'}")
    st.write(f"**Groq API:** {'🟢 Configured' if GROQ_API_KEY else '🟡 Offline Engine'}")
    
    st.caption("To enable live cloud models, set your keys in the `.env` file.")
    st.markdown("---")

    if st.button("🔄 Reset Conversation"):
        st.session_state.session_id = f"session_{int(time.time())}"
        st.session_state.agent = ConversationAgent(mode="rag")
        st.session_state.chat_history = []
        st.rerun()

# Main Header
st.title("🛒 ZentroMart AI Customer Support Agent")
st.markdown(
    "A multi-modal support agent combining **Gnani.ai Speech Models (STT & TTS)**, **RAG Knowledge Retrieval**, "
    "**Caller Identity Verification**, and **Automated Human Escalation**."
)

tab_chat, tab_orders, tab_tickets, tab_kb = st.tabs([
    "💬 Live Voice & Text Agent",
    "📦 Order Lookup & Verification",
    "🎫 Escalation Tickets",
    "📚 Knowledge Base Search"
])

# -------------------------------------------------------------
# TAB 1: LIVE VOICE & TEXT AGENT
# -------------------------------------------------------------
with tab_chat:
    col_chat, col_inspect = st.columns([3, 2])

    with col_chat:
        st.subheader("Interactive Support Channel")
        
        # Display chat messages
        chat_container = st.container(height=420)
        with chat_container:
            if not st.session_state.chat_history:
                st.info("👋 Say hello or upload an audio message to start speaking with ZentroMart Support.")
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.write(msg["text"])
                    if "audio_path" in msg and msg["audio_path"] and Path(msg["audio_path"]).exists():
                        st.audio(msg["audio_path"])

        # Input Area: Audio Upload or Microphone or Text Input
        st.markdown("##### Send Audio or Message")
        audio_file = st.file_uploader(
            "Upload Customer Voice (.wav, .mp3)",
            type=["wav", "mp3", "m4a"],
            help="Customer audio will be transcribed using Gnani STT REST API"
        )
        user_text = st.chat_input("Or type your message here (English, Hindi, Hinglish)...")

        input_message = None
        input_audio_path = None

        if audio_file is not None:
            # Save uploaded audio temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(audio_file.name).suffix) as tmp:
                tmp.write(audio_file.read())
                tmp_path = tmp.name
            
            with st.spinner("🎧 Transcribing with Gnani STT..."):
                transcript = voice_service.transcribe(tmp_path)
                input_message = transcript
                input_audio_path = tmp_path

        elif user_text:
            input_message = user_text

        # Process input through the conversational agent
        if input_message:
            st.session_state.chat_history.append({"role": "user", "text": input_message})

            with st.spinner("🧠 Processing query with ZentroMart Agent..."):
                response = st.session_state.agent.respond(
                    input_message,
                    conversation_id=st.session_state.session_id
                )
                reply_text = response["reply"]
                decision = response.get("decision", "answer")
                evidence = response.get("evidence", [])
                tools_used = response.get("tool_calls", [])

                # Generate TTS voice reply
                ear_text = format_for_speech(reply_text)
                reply_wav_path = OUTPUT_DIR / f"reply_{int(time.time()*1000)}.wav"
                voice_service.synthesize(ear_text, output_file=str(reply_wav_path))

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "text": reply_text,
                    "audio_path": str(reply_wav_path)
                })

                st.session_state.last_response_meta = {
                    "decision": decision,
                    "evidence": evidence,
                    "tool_calls": tools_used,
                    "ear_text": ear_text,
                    "session": response.get("session")
                }
            st.rerun()

    with col_inspect:
        st.subheader("🔍 Real-Time Agent Telemetry")
        if "last_response_meta" in st.session_state:
            meta = st.session_state.last_response_meta
            
            # Decision Badge
            dec = meta["decision"].upper()
            badge_color = {
                "ANSWER": "green",
                "CLARIFY": "blue",
                "ESCALATE": "red",
                "REFUSE": "orange"
            }.get(dec, "gray")
            st.markdown(f"**Current Turn Decision:** :{badge_color}[{dec}]")

            # Verification state
            sess = meta.get("session")
            if sess:
                st.write(f"**Active Order ID:** `{sess.order_id or 'None'}`")
                st.write(f"**Verification Gate:** `{sess.verification_status.upper()}`")
                st.write(f"**Verification Attempts:** `{sess.verification_attempts}`")
                st.write(f"**Ticket Created:** `{'Yes: ' + str(sess.ticket_id) if sess.ticket_created else 'No'}`")

            # Tools Invoked
            st.markdown("---")
            st.markdown("**Tools Executed:**")
            if meta.get("tool_calls"):
                for t in meta["tool_calls"]:
                    st.code(t, language="python")
            else:
                st.caption("No mock tool invocation needed for this turn.")

            # Retrieved Evidence Chunks
            st.markdown("---")
            st.markdown("**Retrieved Knowledge Chunks:**")
            if meta.get("evidence"):
                for sec_id in meta["evidence"]:
                    chunk = rag_engine.get_chunk_by_id(sec_id)
                    with st.expander(f"📌 [{sec_id}] {chunk.title if chunk else ''}"):
                        if chunk:
                            st.caption(f"Doc: {chunk.doc_title} ({chunk.doc_type}) | Status: {chunk.status}")
                            st.write(chunk.content)
            else:
                st.caption("No policy chunks required (clarification / greeting).")
        else:
            st.info("Start chatting to view live agent telemetry, state machine, and retrieved chunks.")

# -------------------------------------------------------------
# TAB 2: ORDER LOOKUP & VERIFICATION
# -------------------------------------------------------------
with tab_orders:
    st.subheader("Order Lookup Simulation (IDV Gated)")
    st.markdown("Simulate the caller identity verification protocol as mandated by `agent_identity_verification.md`.")
    
    col_o1, col_o2 = st.columns(2)
    with col_o1:
        test_oid = st.text_input("Enter 8-digit Order ID", value="89782837")
        test_proof = st.text_input("Enter Verification Proof (Phone Last 4 or Delivery Pincode)", value="")
        check_btn = st.button("Perform Order Lookup")

    with col_o2:
        if check_btn:
            result = lookup_order(test_oid, verification_value=test_proof if test_proof else None)
            if result.get("verified"):
                st.success("✅ Caller Identity Verified! Disclosing order details:")
                st.json(result)
            elif result.get("verification_required"):
                st.warning("🔒 Verification Required! Details are withheld:")
                st.json(result)
            else:
                st.error("❌ Verification Failed or Order Not Found.")
                st.json(result)

# -------------------------------------------------------------
# TAB 3: ESCALATION TICKETS
# -------------------------------------------------------------
with tab_tickets:
    st.subheader("Human Handover & Escalation Tickets")
    tickets_file = PROJECT_ROOT / "Capstone_Project_Dataset_FTTT_2026" / "tools" / "tickets_created.csv"
    if tickets_file.exists():
        try:
            df_t = pd.read_csv(tickets_file)
            st.dataframe(df_t, use_container_width=True)
        except Exception as e:
            st.error(f"Error reading tickets: {e}")
    else:
        st.info("No tickets created yet. When the agent escalates a claim or repeat contact, tickets will appear here.")

# -------------------------------------------------------------
# TAB 4: KNOWLEDGE BASE SEARCH
# -------------------------------------------------------------
with tab_kb:
    st.subheader("Knowledge Base & Catalog Search")
    kb_query = st.text_input("Search policy documents or product catalog:", value="festive sale returns")
    if kb_query:
        results = rag_engine.retrieve(kb_query, top_k=6)
        st.write(f"Found {len(results)} relevant section(s):")
        for chunk, score in results:
            with st.expander(f"[{chunk.section_id}] {chunk.title} (Score: {score:.3f})"):
                st.markdown(f"**Source Document:** `{chunk.source_file}` | **Type:** `{chunk.doc_type}` | **Status:** `{chunk.status}`")
                st.write(chunk.content)
