import pandas as pd
from textblob import TextBlob
import re

KEYWORDS = ["support", "query", "request", "help"]
URGENT_WORDS = ["immediately", "urgent", "critical", "asap", "cannot access", "important"]

def detect_priority(text: str) -> str:
    if not isinstance(text, str):
        return "Normal"
    text_lower = text.lower()
    for word in URGENT_WORDS:
        if word in text_lower:
            return "Urgent"
    return "Normal"

def detect_sentiment(text: str) -> str:
    if not isinstance(text, str):
        return "Neutral"
    analysis = TextBlob(text).sentiment.polarity
    if analysis > 0.1:
        return "Positive"
    elif analysis < -0.1:
        return "Negative"
    return "Neutral"

def fetch_emails(path="data/intern_emails.csv"):
    """Load and filter emails by keywords, auto-tag priority + sentiment."""
    df = pd.read_csv(path)

    # Normalize column names
    df.columns = [c.strip() for c in df.columns]
    lower_map = {c.lower(): c for c in df.columns}

    rename_dict = {}
    if "sender" in lower_map:
        rename_dict[lower_map["sender"]] = "From"
    if "from" in lower_map:
        rename_dict[lower_map["from"]] = "From"
    if "subject" in lower_map:
        rename_dict[lower_map["subject"]] = "Subject"
    if "body" in lower_map:
        rename_dict[lower_map["body"]] = "Body"
    if "date" in lower_map:
        rename_dict[lower_map["date"]] = "Sent Date"
    if "sent_date" in lower_map:
        rename_dict[lower_map["sent_date"]] = "Sent Date"

    if rename_dict:
        df = df.rename(columns=rename_dict)

    # Ensure essential columns
    for col in ["From", "Subject", "Body", "Sent Date"]:
        if col not in df.columns:
            df[col] = ""

    # Filter only relevant emails
    mask = df["Subject"].fillna("").str.lower().str.contains("|".join(KEYWORDS))
    filtered_df = df[mask].copy()

    # Apply sentiment + priority tagging
    filtered_df["Priority"] = filtered_df.apply(
        lambda row: detect_priority(str(row["Subject"]) + " " + str(row["Body"])), axis=1
    )
    filtered_df["Sentiment"] = filtered_df["Body"].apply(detect_sentiment)

    # Reset index
    filtered_df = filtered_df.reset_index(drop=True)
    return filtered_df
