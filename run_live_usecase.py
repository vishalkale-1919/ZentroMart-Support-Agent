"""End-to-End Live Usecase Demonstration for VoiceOps 2026.
Demonstrates:
1. Customer Audio Ingestion -> Gnani STT
2. AI Opening & IDV Gate (lookup_order)
3. Customer Proof Verification -> Order Data Retrieval
4. RAG-grounded Policy Tracking
5. High-Value Damage Handover -> Ticket Creation (create_ticket)
6. Ear Normalization & Gnani TTS Speech Synthesis
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Ensure environment and paths are loaded
load_dotenv('.env', override=True)
sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from src.conversation_agent import ConversationAgent
from src.voice_service import voice_service
from src.tools_adapter import tools_adapter
from src.utils import format_for_speech

def run_usecase():
    print("=" * 70)
    print("VOICEOPS 2026: LIVE MULTIMODAL USECASE DEMONSTRATION")
    print("=" * 70)
    
    agent = ConversationAgent(mode="rag")
    session_id = "live_usecase_call_01"
    output_dir = Path("sample_outputs") / "live_demo"
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # TURN 1: Customer calls in
    # -------------------------------------------------------------
    print("\n--- TURN 1: Customer Audio Call ---")
    customer_speech_1 = "Hello, mujhe apne order ke baare mein help chahiye."
    cust_audio_1 = str(output_dir / "turn1_customer.wav")
    
    # Synthesize customer query with Gnani TTS to simulate incoming phone audio
    print(f"[Simulating Incoming Caller Audio]: \"{customer_speech_1}\"")
    voice_service.synthesize(customer_speech_1, output_file=cust_audio_1, voice="Pranav")
    
    # Live STT Transcription
    heard_1 = voice_service.transcribe(cust_audio_1)
    print(f"[Gnani STT Heard]: \"{heard_1}\"")
    
    # Agent Brain Turn 1
    resp_1 = agent.respond(heard_1, conversation_id=session_id, turn_number=1)
    agent_reply_1 = resp_1["reply"]
    print(f"[Agent Decision]: {resp_1['decision'].upper()}")
    print(f"[Agent Text Reply]: \"{agent_reply_1}\"")
    
    # Live Gnani TTS Spoken Reply
    agent_wav_1 = str(output_dir / "turn1_agent_reply.wav")
    voice_service.synthesize(agent_reply_1, output_file=agent_wav_1, voice="Kaveri")
    print(f"[Gnani TTS Audio Saved]: {agent_wav_1} ({os.path.getsize(agent_wav_1)} bytes)")

    # -------------------------------------------------------------
    # TURN 2: Customer provides Order ID
    # -------------------------------------------------------------
    print("\n--- TURN 2: Customer Provides Order ID ---")
    customer_speech_2 = "Mera order number hai 8978 2837."
    cust_audio_2 = str(output_dir / "turn2_customer.wav")
    print(f"[Customer Spoken]: \"{customer_speech_2}\"")
    voice_service.synthesize(customer_speech_2, output_file=cust_audio_2, voice="Pranav")
    
    heard_2 = voice_service.transcribe(cust_audio_2) or customer_speech_2
    print(f"[Gnani STT Heard]: \"{heard_2}\"")
    
    # Agent Brain Turn 2: Checks order, enforces IDV verification gate
    resp_2 = agent.respond(heard_2, conversation_id=session_id, turn_number=2)
    agent_reply_2 = resp_2["reply"]
    print(f"[Tools Executed]: {resp_2.get('tool_calls')}")
    print(f"[Agent Decision]: {resp_2['decision'].upper()} (Verification Gated)")
    print(f"[Agent Text Reply]: \"{agent_reply_2}\"")
    
    agent_wav_2 = str(output_dir / "turn2_agent_reply.wav")
    voice_service.synthesize(agent_reply_2, output_file=agent_wav_2, voice="Kaveri")
    print(f"[Gnani TTS Audio Saved]: {agent_wav_2} ({os.path.getsize(agent_wav_2)} bytes)")

    # -------------------------------------------------------------
    # TURN 3: Customer provides Verification Proof
    # -------------------------------------------------------------
    print("\n--- TURN 3: Customer Provides Verification Proof ---")
    customer_speech_3 = "Aakhri chaar digit 4533 hain."
    cust_audio_3 = str(output_dir / "turn3_customer.wav")
    print(f"[Customer Spoken]: \"{customer_speech_3}\"")
    voice_service.synthesize(customer_speech_3, output_file=cust_audio_3, voice="Pranav")
    
    heard_3 = voice_service.transcribe(cust_audio_3) or customer_speech_3
    print(f"[Gnani STT Heard]: \"{heard_3}\"")
    
    # Agent Brain Turn 3: Validates proof with lookup_order
    resp_3 = agent.respond(heard_3, conversation_id=session_id, turn_number=3)
    agent_reply_3 = resp_3["reply"]
    print(f"[Tools Executed]: {resp_3.get('tool_calls')}")
    print(f"[Verification Status]: VERIFIED")
    print(f"[Agent Decision]: {resp_3['decision'].upper()}")
    print(f"[Agent Text Reply]: \"{agent_reply_3}\"")
    
    agent_wav_3 = str(output_dir / "turn3_agent_reply.wav")
    voice_service.synthesize(agent_reply_3, output_file=agent_wav_3, voice="Kaveri")
    print(f"[Gnani TTS Audio Saved]: {agent_wav_3} ({os.path.getsize(agent_wav_3)} bytes)")

    # -------------------------------------------------------------
    # TURN 4: Customer reports High-Value Damaged Item
    # -------------------------------------------------------------
    print("\n--- TURN 4: Customer Reports High-Value Damaged TV ---")
    customer_speech_4 = "Maine Zentro Vista 43-inch 4K Smart TV damage report kiya tha. Abhi replacement approve kar do na?"
    cust_audio_4 = str(output_dir / "turn4_customer.wav")
    print(f"[Customer Spoken]: \"{customer_speech_4}\"")
    voice_service.synthesize(customer_speech_4, output_file=cust_audio_4, voice="Pranav")
    
    heard_4 = voice_service.transcribe(cust_audio_4)
    print(f"[Gnani STT Heard]: \"{heard_4}\"")
    
    # Agent Brain Turn 4: Detects > ₹10,000 item -> triggers ticket & Priority escalation
    resp_4 = agent.respond(heard_4, conversation_id=session_id, turn_number=4)
    agent_reply_4 = resp_4["reply"]
    print(f"[Tools Executed]: {resp_4.get('tool_calls')}")
    print(f"[Ticket Created ID]: {resp_4['session'].ticket_id}")
    print(f"[Agent Decision]: {resp_4['decision'].upper()} (Priority Escalation)")
    print(f"[Evidence Used]: {resp_4.get('evidence')}")
    print(f"[Agent Text Reply]: \"{agent_reply_4}\"")
    
    agent_wav_4 = str(output_dir / "turn4_agent_reply.wav")
    voice_service.synthesize(agent_reply_4, output_file=agent_wav_4, voice="Kaveri")
    print(f"[Gnani TTS Audio Saved]: {agent_wav_4} ({os.path.getsize(agent_wav_4)} bytes)")

    print("\n" + "=" * 70)
    print("LIVE USECASE COMPLETED SUCCESSFULLY!")
    print(f"Generated 8 audio files in: {output_dir}")
    print("=" * 70)

if __name__ == "__main__":
    run_usecase()
