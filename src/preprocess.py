# src/preprocess.py

import re
import pandas as pd

def preprocess_emails(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess emails and extract useful fields:
    - Normalize headers (handles variations like 'sender'/'from')
    - Priority (heuristic)
    - Sentiment (heuristic)
    - Phone numbers
    - Sender name
    - Requirement keywords
    - Preview text (first 50 chars of body)
    """

    # ------------------------------
    # Step 1: Normalize headers
    # ------------------------------
    cols_map = {c: c.strip() for c in df.columns}
    df = df.rename(columns=cols_map)

    lower_map = {c.lower().strip(): c for c in df.columns}

    rename_dict = {}
    if "sender" in lower_map:
        rename_dict[lower_map["sender"]] = "From"
    if "from" in lower_map:
        rename_dict[lower_map["from"]] = "From"
    if "subject" in lower_map:
        rename_dict[lower_map["subject"]] = "Subject"
    if "body" in lower_map:
        rename_dict[lower_map["body"]] = "Body"
    if "content" in lower_map:
        rename_dict[lower_map["content"]] = "Body"
    if "sent_date" in lower_map:
        rename_dict[lower_map["sent_date"]] = "Sent Date"

    if rename_dict:
        df = df.rename(columns=rename_dict)

    # Ensure base columns exist
    for col in ["From", "Subject", "Body"]:
        if col not in df.columns:
            df[col] = ""

    # ------------------------------
    # Step 2: Priority detection
    # ------------------------------
    def heuristic_priority(text):
        if not isinstance(text, str):
            return "Normal"
        text_l = text.lower()
        if any(word in text_l for word in ["urgent", "asap", "immediately", "critical", "cannot access"]):
            return "Urgent"
        elif any(word in text_l for word in ["reminder", "follow up", "pending"]):
            return "High"
        return "Normal"

    # ------------------------------
    # Step 3: Sentiment detection
    # ------------------------------
    def heuristic_sentiment(text):
        if not isinstance(text, str):
            return "Neutral"
        text_l = text.lower()
        if any(w in text_l for w in ["thank", "thanks", "great", "appreciate", "good job"]):
            return "Positive"
        if any(w in text_l for w in ["bad", "complaint", "delay", "problem", "issue", "not working"]):
            return "Negative"
        return "Neutral"

    # ------------------------------
    # Step 4: Phone number extraction
    # ------------------------------
    phone_pattern = r"(\+?\d[\d\s\-]{7,}\d)"
    df["Phone"] = df.apply(
        lambda row: re.findall(phone_pattern, str(row.get("Body", "")) + " " + str(row.get("Content", ""))),
        axis=1
    )

    # ------------------------------
    # Step 5: Sender name extraction (fix)
    # ------------------------------
    if "From" in df.columns and df["From"].notna().any():
        df["SenderName"] = df["From"].apply(
            lambda s: str(s).split("@")[0].replace(".", " ").title()
            if pd.notna(s) and "@" in str(s)
            else "Unknown"
        )
    elif "Sender" in df.columns and df["Sender"].notna().any():
        df["SenderName"] = df["Sender"].apply(
            lambda s: str(s).split("@")[0].replace(".", " ").title()
            if pd.notna(s) and "@" in str(s)
            else "Unknown"
        )
    else:
        df["SenderName"] = "Unknown"

    # ------------------------------
    # Step 6: Requirement extraction
    # ------------------------------
    def extract_requirement(text):
        if not isinstance(text, str):
            return "General"
        text_l = text.lower()
        if "password" in text_l or "login" in text_l:
            return "Account Access"
        if "payment" in text_l or "invoice" in text_l or "billing" in text_l:
            return "Billing"
        if "error" in text_l or "bug" in text_l or "not working" in text_l:
            return "Technical Issue"
        if "support" in text_l or "help" in text_l or "query" in text_l or "request" in text_l:
            return "Support Request"
        return "General"

    df["Requirement"] = df["Body"].apply(extract_requirement)

    # ------------------------------
    # Step 7: Preview snippet
    # ------------------------------
    df["Preview"] = df["Body"].apply(
        lambda x: str(x)[:50] + "..." if isinstance(x, str) and len(x) > 50 else str(x)
    )

    # ------------------------------
    # Final Columns
    # ------------------------------
    if "Priority" not in df.columns:
        df["Priority"] = df["Body"].apply(heuristic_priority)

    if "Sentiment" not in df.columns:
        df["Sentiment"] = df["Body"].apply(heuristic_sentiment)

    return df.reset_index(drop=True)
