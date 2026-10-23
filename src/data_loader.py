"""Data loader module for ZentroMart CSV tables."""
import pandas as pd
from typing import Dict, List, Optional
from src.config import DATA_DIR, PRACTICE_DIR

class DataLoader:
    def __init__(self):
        self._customers_df: Optional[pd.DataFrame] = None
        self._orders_df: Optional[pd.DataFrame] = None
        self._order_items_df: Optional[pd.DataFrame] = None
        self._products_df: Optional[pd.DataFrame] = None
        self._pincodes_df: Optional[pd.DataFrame] = None
        self._holidays_df: Optional[pd.DataFrame] = None
        self._support_logs_df: Optional[pd.DataFrame] = None
        
        # In-memory dictionaries for instant keyed lookups
        self.customers: Dict[str, dict] = {}
        self.orders: Dict[str, dict] = {}
        self.order_items: Dict[str, List[dict]] = {}
        self.products: Dict[str, dict] = {}
        self.pincodes: Dict[str, dict] = {}

        self.load_all()

    def load_all(self):
        """Load all CSV tables with string dtype to preserve IDs and pincodes."""
        if (DATA_DIR / "customers.csv").exists():
            self._customers_df = pd.read_csv(DATA_DIR / "customers.csv", dtype=str)
            for row in self._customers_df.to_dict(orient="records"):
                self.customers[row["customer_id"]] = row

        if (DATA_DIR / "orders.csv").exists():
            self._orders_df = pd.read_csv(DATA_DIR / "orders.csv", dtype=str)
            for row in self._orders_df.to_dict(orient="records"):
                self.orders[row["order_id"]] = row

        if (DATA_DIR / "order_items.csv").exists():
            self._order_items_df = pd.read_csv(DATA_DIR / "order_items.csv", dtype=str)
            for row in self._order_items_df.to_dict(orient="records"):
                oid = row["order_id"]
                if oid not in self.order_items:
                    self.order_items[oid] = []
                self.order_items[oid].append(row)

        if (DATA_DIR / "products.csv").exists():
            self._products_df = pd.read_csv(DATA_DIR / "products.csv", dtype=str)
            for row in self._products_df.to_dict(orient="records"):
                self.products[row["sku"]] = row

        if (DATA_DIR / "pincodes.csv").exists():
            self._pincodes_df = pd.read_csv(DATA_DIR / "pincodes.csv", dtype=str)
            for row in self._pincodes_df.to_dict(orient="records"):
                self.pincodes[row["pincode"]] = row

    def get_product(self, sku: str) -> Optional[dict]:
        return self.products.get(sku.strip())

    def get_customer(self, customer_id: str) -> Optional[dict]:
        return self.customers.get(customer_id.strip())

    def get_order(self, order_id: str) -> Optional[dict]:
        clean_id = ''.join(ch for ch in str(order_id) if ch.isdigit())
        return self.orders.get(clean_id)

    def get_order_items(self, order_id: str) -> List[dict]:
        clean_id = ''.join(ch for ch in str(order_id) if ch.isdigit())
        return self.order_items.get(clean_id, [])

    def get_pincode_info(self, pincode: str) -> Optional[dict]:
        return self.pincodes.get(str(pincode).strip())

    @staticmethod
    def load_practice_questions() -> pd.DataFrame:
        path = PRACTICE_DIR / "questions.csv"
        return pd.read_csv(path, dtype=str)

    @staticmethod
    def load_practice_conversations() -> pd.DataFrame:
        path = PRACTICE_DIR / "conversations.csv"
        return pd.read_csv(path, dtype=str)

# Global singleton loader
data_loader = DataLoader()
