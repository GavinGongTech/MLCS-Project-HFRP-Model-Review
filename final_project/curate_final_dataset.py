import pandas as pd

# Load text + risk dataset
text_df = pd.read_csv("financial_text_dataset_with_risk.csv")

print("Loaded financial_text_dataset_with_risk.csv with shape:", text_df.shape)
print("Columns:", text_df.columns.tolist())

# Rename CIK-style ticker to 'cik', and yf_ticker (true symbol) to 'ticker'; just for formatting issues
if "ticker" in text_df.columns:
    text_df = text_df.rename(columns={"ticker": "cik"})
if "yf_ticker" in text_df.columns:
    text_df = text_df.rename(columns={"yf_ticker": "ticker"})

# Ensure fiscal_year exists
if "fiscal_year" not in text_df.columns:
    if "period_of_report" in text_df.columns:
        date_col = "period_of_report"
    else:
        date_col = "filing_date"
    text_df["fiscal_year"] = pd.to_datetime(text_df[date_col]).dt.year

# We want to only keep rows with non-missing ticker and future_vol here
text_df = text_df.dropna(subset=["ticker", "future_vol"])

print("After dropping rows with missing ticker/future_vol:", text_df.shape)

# The next step is to load the numeric financials dataset
funds = pd.read_csv("structured_financials.csv")
print("Loaded structured_financials.csv with shape:", funds.shape)
print("Numeric columns:", funds.columns.tolist())

# We can also drop the rows with missing key numeric features
funds = funds.dropna(subset=["revenue", "net_income", "eps"])
print("After dropping rows with missing revenue/net_income/eps:", funds.shape)

# To replicate the step in the paper of fusion, we need to merge the text + risk dataset
# with the numeric financials dataset on (ticker, fiscal_year)
merged = ( # Simple inner join on ticker + fiscal_year
    text_df
    .merge(
        funds,
        on=["ticker", "fiscal_year"],
        how="inner",
        suffixes=("_text", "_fin"),
    )
)

print("Merged dataset shape:", merged.shape) # shape after merging text + numeric on (ticker, fiscal_year)
# We need to select the final columns we want in the dataset; these include:
cols_final = [
    "company",
    "ticker",
    "cik",
    "fiscal_year",
    "filing_date",
    "period_of_report",
    "revenue", # These are the numeric features (remember we want to merge them together)
    "net_income",
    "operating_income",
    "eps",
    "total_assets",
    "total_liabilities",
    "cfo",
    "textual_disclosures", # This is the combined text field
    "future_vol", # This is our target variable (numeric risk label); remember we are running a risk regresion to obtain this
]

cols_final = [c for c in cols_final if c in merged.columns]
dataset = merged[cols_final]

# Save the final dataset
dataset.to_csv("hfrp_dataset.csv", index=False)
print("Final dataset shape:", dataset.shape)
print("Saved to hfrp_dataset.csv")

print("\nPreview:")
print(dataset.head())
