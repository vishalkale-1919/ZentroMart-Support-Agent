"""Utility functions: date math, text normalization for voice, and formatting."""
import re
from datetime import date, timedelta
from typing import Set, Optional
from src.config import REFERENCE_DATE, DATA_DIR
import pandas as pd

# Load holidays from data/holidays.csv
def load_holidays() -> Set[date]:
    holidays_file = DATA_DIR / "holidays.csv"
    holidays = set()
    if holidays_file.exists():
        try:
            df = pd.read_csv(holidays_file)
            for d_str in df['date'].dropna():
                holidays.add(date.fromisoformat(str(d_str).strip()))
        except Exception:
            pass
    # Fallback to known holidays if file load fails
    if not holidays:
        holidays = {
            date(2026, 8, 15),
            date(2026, 10, 2),
            date(2026, 10, 19),
            date(2026, 11, 9),
        }
    return holidays

HOLIDAYS = load_holidays()

def is_business_day(d: date, holidays: Optional[Set[date]] = None) -> bool:
    """Check if date is Monday-Saturday and not a holiday."""
    if holidays is None:
        holidays = HOLIDAYS
    # Monday is 0, Saturday is 5, Sunday is 6
    if d.weekday() == 6:  # Sunday is non-business day
        return False
    if d in holidays:
        return False
    return True

def add_business_days(start_date: date, n_days: int, holidays: Optional[Set[date]] = None) -> date:
    """Add n business days (Mon-Sat, skipping holidays) to start_date."""
    if holidays is None:
        holidays = HOLIDAYS
    curr = start_date
    added = 0
    while added < n_days:
        curr += timedelta(days=1)
        if is_business_day(curr, holidays):
            added += 1
    return curr

def count_business_days_between(start_date: date, end_date: date, holidays: Optional[Set[date]] = None) -> int:
    """Count business days strictly between start_date and end_date (inclusive of end_date)."""
    if holidays is None:
        holidays = HOLIDAYS
    if start_date >= end_date:
        return 0
    curr = start_date + timedelta(days=1)
    count = 0
    while curr <= end_date:
        if is_business_day(curr, holidays):
            count += 1
        curr += timedelta(days=1)
    return count

def format_for_speech(text: str) -> str:
    """Normalize written text for speech synthesis (ear-friendly).
    
    Rules from PDF & Style guide [STY-3]:
    - Read order IDs in groups of digits with spaces
    - Say amounts in rupees (e.g., '₹49' -> '49 rupees')
    - Format dates with month names ('the 24th of October 2026')
    - Strip markdown markers (**, *, #, _, `)
    """
    if not text:
        return ""
    
    # Strip markdown syntax
    cleaned = re.sub(r'[*_#`]', '', text)
    cleaned = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', cleaned)  # remove markdown links
    cleaned = re.sub(r'\[[A-Z0-9\.\-]+\]', '', cleaned)           # remove [SEC-ID] tags

    # Convert currency symbols
    cleaned = re.sub(r'₹\s*(\d+(?:,\d+)*)', r'\1 rupees', cleaned)
    cleaned = re.sub(r'(?:Rs\.?|INR)\s*(\d+(?:,\d+)*)', r'\1 rupees', cleaned)

    # Convert 8-digit order IDs into spaced digit groups: e.g. "89782837" -> "8 9 7 8, 2 8 3 7"
    def _split_order_id(match):
        digits = match.group(0)
        return f"{digits[:4]} {digits[4:]}"
    
    cleaned = re.sub(r'\b\d{8}\b', _split_order_id, cleaned)

    # Normalize dates if found in YYYY-MM-DD
    month_names = {
        1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
        7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"
    }
    def _date_repl(match):
        y, m, d = int(match.group(1)), int(match.group(2)), int(match.group(3))
        m_name = month_names.get(m, "")
        return f"the {d} of {m_name} {y}"
    cleaned = re.sub(r'\b(202[4-9])-0?([1-9]|1[0-2])-0?([1-9]|[12]\d|3[01])\b', _date_repl, cleaned)

    # Clean whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

WORD_MAP = {
    'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
    'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
    'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
    'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19,
    'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50,
    'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90
}

def parse_spoken_number(phrase: str) -> str:
    words = phrase.lower().replace('-', ' ').split()
    total = 0
    current = 0
    for w in words:
        if w in WORD_MAP:
            current += WORD_MAP[w]
        elif w == 'hundred':
            current = (current if current else 1) * 100
        elif w == 'thousand':
            total += (current if current else 1) * 1000
            current = 0
    total += current
    return str(total)

def extract_spoken_digits(text: str, target_len: Optional[int] = None) -> str:
    """Extract 4, 6, or 8 digits from speech text (handles numerals and words)."""
    # Remove metadata counting phrases such as 'aakhri chaar digit', 'last 4 digits', 'last four digits'
    cleaned_text = re.sub(
        r'(?:last\s*(?:4|four)?\s*digits?|aakhri\s*(?:4|chaar|char)?\s*(?:digit|ank)?|chaar\s*digit|4\s*digits?|pincode|pin\s*code)',
        '',
        text,
        flags=re.I
    ).strip()

    # 1. Direct digits
    digits = ''.join(ch for ch in cleaned_text if ch.isdigit())
    if target_len and len(digits) == target_len:
        return digits
    if len(digits) in (4, 6, 8):
        return digits

    # 2. Check for sequence of single digit words
    words = cleaned_text.lower().replace('-', ' ').split()
    single_digits = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'shunya': '0', 'ek': '1', 'do': '2', 'teen': '3', 'chaar': '4',
        'paanch': '5', 'chhe': '6', 'saat': '7', 'aath': '8', 'nau': '9'
    }
    seq = [single_digits[w] for w in words if w in single_digits]
    if target_len and len(seq) == target_len:
        return ''.join(seq)
    if len(seq) in (4, 6, 8):
        return ''.join(seq)

    # 3. Compound number parse (e.g. four thousand five hundred thirty three)
    val = parse_spoken_number(cleaned_text)
    if target_len and len(val) == target_len:
        return val
    if len(val) in (4, 6, 8):
        return val

    return digits

