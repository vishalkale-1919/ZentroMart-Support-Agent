#!/usr/bin/env python3
"""VoiceOps 2026 Batch Runner.

Command required by organizers:
python run.py --input-dir <audio_folder> --output-dir <output_folder>

Takes a folder of customer audio files, runs the ZentroMart agent on each,
and saves the replies (as text and audio) with the document chunks used.
"""
import os
import sys
import argparse
import json
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.conversation_agent import ConversationAgent
from src.voice_service import voice_service
from src.utils import format_for_speech

AUDIO_EXTENSIONS = {'.wav', '.mp3', '.m4a', '.ogg', '.flac'}
TEXT_EXTENSIONS = {'.txt'}

def parse_args():
    parser = argparse.ArgumentParser(description="VoiceOps 2026 Batch Agent Runner")
    parser.add_argument(
        "--input-dir", "-i",
        type=str,
        default="./sample_inputs",
        help="Path to folder containing customer audio files (.wav, .mp3, etc.) or .txt questions"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./sample_outputs",
        help="Path to folder where replies, audio, and chunk JSONs will be saved"
    )
    parser.add_argument(
        "--language", "-l",
        type=str,
        default="en-IN",
        help="Language code for Gnani STT/TTS (e.g., en-IN, hi-IN, hi-en)"
    )
    return parser.parse_args()

def run_batch():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist.")
        print(f"Creating '{input_dir}' with sample test files...")
        input_dir.mkdir(parents=True, exist_ok=True)
        # Create a sample question text file
        (input_dir / "sample_01.txt").write_text(
            "How fast are refunds to the ZentroMart Wallet?",
            encoding="utf-8"
        )

    # Collect files
    input_files = sorted([
        f for f in input_dir.iterdir()
        if f.is_file() and (f.suffix.lower() in AUDIO_EXTENSIONS or f.suffix.lower() in TEXT_EXTENSIONS)
    ])

    if not input_files:
        print(f"No audio or text files found in {input_dir}.")
        print("Supported formats:", list(AUDIO_EXTENSIONS | TEXT_EXTENSIONS))
        return 0

    print("=" * 65)
    print("VoiceOps 2026 - ZentroMart Customer Support Agent")
    print(f"Processing {len(input_files)} input file(s) from: {input_dir}")
    print(f"Saving outputs to: {output_dir}")
    print("=" * 65)

    batch_summary = []
    agent = ConversationAgent(mode="rag")

    for idx, file_path in enumerate(input_files, 1):
        start_time = time.time()
        file_stem = file_path.stem
        is_audio = file_path.suffix.lower() in AUDIO_EXTENSIONS

        print(f"\n[{idx}/{len(input_files)}] Processing: {file_path.name}")

        # 1. Speech-to-Text / Input Ingestion
        if is_audio:
            transcript = voice_service.transcribe(str(file_path), language_code=args.language)
            print(f"  [STT Transcript]: \"{transcript}\"")
        else:
            transcript = file_path.read_text(encoding="utf-8").strip()
            print(f"  [Text Input]: \"{transcript}\"")

        # Handle unclear / empty audio
        if not transcript.strip():
            agent_reply = "Sorry, I didn't catch that. Could you say it again?"
            evidence = []
            chunks_used = []
            decision = "clarify"
        else:
            # 2. Conversational Agent Brain (RAG + Context + Tools)
            # Use file_stem as session identifier
            resp = agent.respond(transcript, conversation_id=file_stem)
            agent_reply = resp["reply"]
            evidence = resp.get("evidence", [])
            chunks_used = resp.get("session").history if resp.get("session") else []
            decision = resp.get("decision", "answer")

        print(f"  [Agent Decision]: {decision}")
        print(f"  [Agent Reply]: \"{agent_reply}\"")
        print(f"  [Evidence / Chunks]: {evidence}")

        # 3. Text Normalization for Voice & TTS Synthesis
        ear_text = format_for_speech(agent_reply)
        out_wav_path = output_dir / f"{file_stem}_reply.wav"
        voice_service.synthesize(
            text=ear_text,
            output_file=str(out_wav_path),
            language=args.language
        )
        print(f"  [TTS Audio Output]: {out_wav_path.name} ({out_wav_path.stat().st_size} bytes)")

        # 4. Save Text Outputs
        transcript_path = output_dir / f"{file_stem}_transcript.txt"
        transcript_path.write_text(transcript, encoding="utf-8")

        reply_path = output_dir / f"{file_stem}_reply.txt"
        reply_path.write_text(agent_reply, encoding="utf-8")

        chunks_path = output_dir / f"{file_stem}_chunks.json"
        chunks_data = {
            "file": file_path.name,
            "transcript": transcript,
            "decision": decision,
            "evidence_section_ids": evidence,
            "agent_reply": agent_reply,
            "spoken_ear_text": ear_text,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "processing_seconds": round(time.time() - start_time, 2)
        }
        chunks_path.write_text(json.dumps(chunks_data, indent=2, ensure_ascii=False), encoding="utf-8")

        batch_summary.append(chunks_data)

    # 5. Save Summary Report
    summary_path = output_dir / "batch_summary.json"
    summary_path.write_text(json.dumps(batch_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n" + "=" * 65)
    print(f"Batch processing complete! Processed {len(input_files)} file(s).")
    print(f"Summary saved to: {summary_path}")
    print("=" * 65)
    return 0

if __name__ == "__main__":
    sys.exit(run_batch())
