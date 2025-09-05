# app.py
import streamlit as st
import pandas as pd
import datetime
import re
from src import db_helper


from src import preprocess, reply_generator, analytics
# Initialize SQLite DB (creates table if not exists)
db_helper.init_db()
# ---------------------------
# Helpers
# ---------------------------
FILTER_KEYWORDS = ["support", "query", "request", "help"]

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize and standardize email dataset headers."""
    cols_map = {c: c.strip() for c in df.columns}
    df = df.rename(columns=cols_map)
    lower_map = {c.lower().strip(): c for c in df.columns}

    rename_dict = {}
    if "sender" in lower_map: rename_dict[lower_map["sender"]] = "From"
    if "from" in lower_map: rename_dict[lower_map["from"]] = "From"
    if "subject" in lower_map: rename_dict[lower_map["subject"]] = "Subject"
    if "body" in lower_map: rename_dict[lower_map["body"]] = "Body"
    if "sent_date" in lower_map: rename_dict[lower_map["sent_date"]] = "Sent Date"
    if "sentdate" in lower_map: rename_dict[lower_map["sentdate"]] = "Sent Date"

    if rename_dict:
        df = df.rename(columns=rename_dict)

    for col in ["From", "Subject", "Body"]:
        if col not in df.columns:
            df[col] = ""

    if "Sent Date" in df.columns:
        try:
            df["Sent Date"] = pd.to_datetime(df["Sent Date"], errors="coerce")
        except Exception:
            pass

    return df

def ensure_metadata_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure Priority and Sentiment exist."""
    if "Priority" not in df.columns:
        df["Priority"] = "Normal"
    if "Sentiment" not in df.columns:
        df["Sentiment"] = "Neutral"
    return df

def filter_support_emails(df: pd.DataFrame) -> pd.DataFrame:
    """Filter emails with support-related subjects."""
    if "Subject" not in df.columns:
        return df
    mask = df["Subject"].fillna("").str.lower().str.contains("|".join(FILTER_KEYWORDS))
    return df[mask].reset_index(drop=True)

def classify_priority(text: str) -> str:
    urgent_keywords = ["immediately", "urgent", "critical", "cannot access", "asap", "as soon as possible", "now"]
    t = (text or "").lower()
    return "Urgent" if any(k in t for k in urgent_keywords) else "Normal"

def classify_sentiment(text: str) -> str:
    t = (text or "").lower()
    negative_words = ["angry", "frustrated", "disappointed", "not happy", "hate", "bad", "terrible", "worst"]
    positive_words = ["thank you", "great", "happy", "love", "excellent", "thanks"]
    if any(w in t for w in negative_words):
        return "Negative"
    if any(w in t for w in positive_words):
        return "Positive"
    return "Neutral"

def extract_info(row) -> dict:
    # Check both Body and Content
    t = (row.get("Body", "") or "") + " " + (row.get("Content", "") or "")
    phone = re.findall(r"\+?\d[\d\-\s]{7,}\d", t)
    email = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", t)
    return {"Phone": phone[0] if phone else "", "AltEmail": email[0] if email else ""}

# ---------------------------
# Streamlit Page Setup
# ---------------------------
st.set_page_config(page_title="AI Communication Assistant", layout="wide")
st.title("📧 AI-Powered Communication Assistant")

# ---------------------------
# Load dataset (SQLite integration)
# ---------------------------
@st.cache_data
def load_data(path="data/intern_emails.csv"):
    try:
        # Attempt to load from DB first
        df = db_helper.load_emails()
        if df.empty:
            # If DB empty, fallback to CSV
            df = pd.read_csv(path)
            db_helper.save_emails(df)
    except Exception:
        # If DB fails, fallback to CSV
        df = pd.read_csv(path)
        db_helper.save_emails(df)
    return df

try:
    df = load_data()
    df = normalize_columns(df)
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

try:
    processed_df = preprocess.preprocess_emails(df.copy())
except Exception as e:
    st.warning("⚠️ Preprocessing failed — using normalized raw data instead.")
    st.write(f"Preprocess error: {e}")
    processed_df = df.copy()


# ---------------------------
# Metadata enrichment
# ---------------------------
processed_df = ensure_metadata_columns(processed_df)
processed_df["Priority"] = processed_df["Body"].apply(classify_priority)
processed_df["Sentiment"] = processed_df["Body"].apply(classify_sentiment)

extracted = processed_df.apply(extract_info, axis=1)
processed_df["Phone"] = extracted.apply(lambda x: x["Phone"])
processed_df["AltEmail"] = extracted.apply(lambda x: x["AltEmail"])

processed_df = filter_support_emails(processed_df)

# Sort urgent first
priority_order = {"Urgent": 1, "Normal": 0}
processed_df["PriorityRank"] = processed_df["Priority"].map(priority_order).fillna(0)
processed_df = processed_df.sort_values(by="PriorityRank", ascending=False).reset_index(drop=True)
processed_df.drop(columns=["PriorityRank"], inplace=True)

# ---------------------------
# Tabs
# ---------------------------
tab1, tab2 = st.tabs(["📥 Inbox", "📊 Analytics"])

# ---------------------------
# Inbox Tab
# ---------------------------
with tab1:
    st.subheader("Inbox")
    st.caption("Detected columns: " + ", ".join(processed_df.columns))

    preferred = ["Subject", "Priority", "Sentiment", "From", "Phone", "AltEmail"]
    display_cols = [c for c in preferred if c in processed_df.columns] or processed_df.columns[:4].tolist()

    st.dataframe(processed_df[display_cols], use_container_width=True)

    def format_subject(idx):
        subj = processed_df.loc[idx, "Subject"] if "Subject" in processed_df.columns else ""
        sender = processed_df.loc[idx, "From"] if "From" in processed_df.columns else ""
        snippet = (subj[:60] + "...") if isinstance(subj, str) and len(subj) > 60 else subj
        return f"{idx} — {snippet}  ({sender})"

    email_idx = st.selectbox("Select an email:", processed_df.index, format_func=format_subject)

    if email_idx is not None:
        email = processed_df.loc[email_idx]
        st.markdown(f"### 📌 Subject: {email.get('Subject', '(no subject)')}")
        st.markdown(f"**From:** {email.get('From', '')}")
        st.markdown(f"**Phone:** {email.get('Phone','')}")
        st.markdown(f"**Alt Email:** {email.get('AltEmail','')}")
        st.text_area("Email body", value=email.get("Body", ""), height=200)
        st.markdown(f"**Priority:** {email.get('Priority','')} | **Sentiment:** {email.get('Sentiment','')}")

        if "draft_replies" not in st.session_state:
            st.session_state.draft_replies = {}

        if st.button("Generate Reply"):
            with st.spinner("Generating reply..."):
                try:
                    reply = reply_generator.generate_reply(
                        subject=email.get("Subject", ""),
                        body=email.get("Body", ""),
                        sentiment=email.get("Sentiment", "Neutral"),
                        priority=email.get("Priority", "Normal")
                    )
                except Exception as e:
                    reply = f"(Reply generation failed: {e})"
                st.session_state.draft_replies[email_idx] = reply

        draft = st.session_state.draft_replies.get(email_idx, "")
        st.text_area("✍️ Draft Reply", value=draft, height=200, key=f"draft_{email_idx}")

# ---------------------------
# Analytics Tab
# ---------------------------
with tab2:
    st.subheader("Email Analytics")
    try:
        stats = analytics.get_stats(processed_df)
        st.metric("Total Emails", stats["Total Emails"])
        st.metric("Last 24h", stats["Last 24h"])
        st.metric("Urgent", stats["Urgent"])
        st.metric("High Priority", stats["High Priority"])
        st.metric("Normal Priority", stats["Normal"])
        st.metric("Positive", stats["Positive"])
        st.metric("Negative", stats["Negative"])
        st.metric("Neutral", stats["Neutral"])

        # Show enhanced charts
        analytics.show_charts(processed_df)

    except Exception as e:
        st.warning("Analytics module raised an error or returned nothing.")
        st.write(f"Analytics error: {e}")
