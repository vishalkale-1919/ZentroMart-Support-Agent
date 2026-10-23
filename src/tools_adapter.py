"""Tools adapter module: wraps ZentroMart identity-checked tools."""
import sys
import os
from typing import Dict, Any, Optional
from src.config import TOOLS_DIR

# Ensure Capstone_Project_Dataset_FTTT_2026 is on sys.path
parent_dir = str(TOOLS_DIR.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from tools.tools import lookup_order as _raw_lookup_order
    from tools.tools import create_ticket as _raw_create_ticket
except ImportError:
    # If tools is in root
    from tools import lookup_order as _raw_lookup_order
    from tools import create_ticket as _raw_create_ticket

class ToolsAdapter:
    def __init__(self):
        self.call_history = []

    def lookup_order(self, order_id: str, verification_value: Optional[str] = None) -> Dict[str, Any]:
        """Perform order lookup with identity verification gating.
        
        If verification_value is omitted/empty: returns verification prompt.
        If provided: validates against 4-digit phone suffix or 6-digit delivery pincode.
        """
        result = _raw_lookup_order(order_id=order_id, verification_value=verification_value)
        self.call_history.append({
            "tool": "lookup_order",
            "order_id": order_id,
            "verification_value": verification_value,
            "result_summary": {
                "verified": result.get("verified", False),
                "verification_required": result.get("verification_required", False),
                "found": result.get("found", False)
            }
        })
        return result

    def create_ticket(self, summary: str, order_id: Optional[str] = None,
                      reason: Optional[str] = None, intent: Optional[str] = None) -> Dict[str, Any]:
        """Create an escalation ticket. Only valid if summary is >= 20 characters."""
        result = _raw_create_ticket(
            summary=summary,
            order_id=order_id,
            reason=reason,
            intent=intent
        )
        self.call_history.append({
            "tool": "create_ticket",
            "summary": summary,
            "order_id": order_id,
            "reason": reason,
            "intent": intent,
            "result": result
        })
        return result

# Singleton adapter
tools_adapter = ToolsAdapter()
lookup_order = tools_adapter.lookup_order
create_ticket = tools_adapter.create_ticket
