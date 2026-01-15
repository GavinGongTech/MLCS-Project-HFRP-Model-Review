# json_to_text.py
from pathlib import Path
import pandas as pd
import json

PROJECT_ROOT = Path(__file__).resolve().parent
json_dir = PROJECT_ROOT / "edgar_crawler" / "datasets" / "EXTRACTED_FILINGS"

print("Looking for JSONs in:", json_dir.resolve())
print("Directory exists?", json_dir.exists())

rows = []
for path in json_dir.rglob("*.json"):
    with open(path, "r") as f:
        doc = json.load(f)
    rows.append({
        "ticker": doc.get("ticker") or doc.get("symbol") or doc.get("cik"),
        "company": doc.get("company"),
        "filing_date": doc.get("filing_date"),
        "period_of_report": doc.get("period_of_report"),
        "risk_text": doc.get("item_1A", "") or "",
        "mdna_text": doc.get("item_7", "") or "",
    })

print("Number of JSON filings loaded:", len(rows))

text_df = pd.DataFrame(rows)
print("Columns in text_df:", list(text_df.columns))

# Combine risk + MD&A into one field used by the model
text_df["textual_disclosures"] = (
    text_df["risk_text"].fillna("") + "\n\n" + text_df["mdna_text"].fillna("")
)

# Add fiscal year for merging later
date_col = "period_of_report" if "period_of_report" in text_df.columns else "filing_date"
text_df["fiscal_year"] = pd.to_datetime(text_df[date_col]).dt.year

out_path = PROJECT_ROOT / "financial_text_dataset.csv"
text_df.to_csv(out_path, index=False)
print("Saved text dataset to:", out_path, "with shape", text_df.shape)
