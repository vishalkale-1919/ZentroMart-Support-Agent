"""Configuration module for VoiceOps 2026."""
import os
from pathlib import Path
from datetime import date
from dotenv import load_dotenv

# Base paths
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

# Load .env file (with override=True to guarantee latest keys/models take precedence)
load_dotenv(PROJECT_ROOT / ".env", override=True)

# Dataset locations
DATASET_DIR = PROJECT_ROOT / "Capstone_Project_Dataset_FTTT_2026"
if not DATASET_DIR.exists():
    # Fallback to local project root if files are directly at root
    DATASET_DIR = PROJECT_ROOT

KB_DIR = DATASET_DIR / "kb"
DATA_DIR = DATASET_DIR / "data"
PRACTICE_DIR = DATASET_DIR / "practice"
TOOLS_DIR = DATASET_DIR / "tools"

# Reference Date for ZentroMart simulation
# Frozen date as specified in brief: 20 October 2026
REFERENCE_DATE_STR = os.getenv("REFERENCE_DATE", "2026-10-20")
try:
    REFERENCE_DATE = date.fromisoformat(REFERENCE_DATE_STR)
except Exception:
    REFERENCE_DATE = date(2026, 10, 20)

# API Keys
GNANI_API_KEY = os.getenv("GNANI_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

# LLM Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower().strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b").strip()

# Gnani Voice Configuration
GNANI_TTS_MODEL = os.getenv("GNANI_TTS_MODEL", "timbre-v2.5").strip()
GNANI_TTS_VOICE = os.getenv("GNANI_TTS_VOICE", "Kaveri").strip()
GNANI_DEFAULT_LANG = os.getenv("GNANI_DEFAULT_LANG", "en-IN").strip()

# Output directories
OUTPUT_DIR = PROJECT_ROOT / "sample_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
