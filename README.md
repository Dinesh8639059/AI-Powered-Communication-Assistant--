AI-Powered Communication Assistant
Overview

The AI-Powered Communication Assistant is a full-stack email management platform that helps users automatically process, categorize, and respond to incoming emails. It integrates Natural Language Processing (NLP) heuristics and a Retrieval-Augmented Generation (RAG) model to provide AI-powered replies.

This project was developed as a practical solution to handle large volumes of emails efficiently, prioritizing urgent messages, classifying sentiment, and extracting key details like sender name, phone number, and alternate email.

Features
Inbox Management

Load emails from CSV or database.

Normalize email headers (handles variations like sender / from / body / content).

Filter emails for support queries and requests.

Display email details: Subject, From, Body, Date, Priority, Sentiment, Phone, AltEmail.

Preview snippet of email content.

AI-Powered Reply Generation

Uses a RAG approach: Retrieves context from a knowledge base and generates replies using Groq’s LLM.

Replies consider:

Email sentiment (Positive / Negative / Neutral)

Email priority (Normal / Urgent)

Knowledge base context

Draft replies can be viewed and edited before sending.

Email Metadata Extraction

Priority classification (Urgent, High, Normal) based on heuristics.

Sentiment analysis (Positive, Negative, Neutral) based on heuristics.

Sender Name Extraction: Parses email addresses to detect names.

Phone Number & Alternate Email: Extracted from email body automatically.

Requirement Type: Classifies emails into categories (Account Access, Billing, Technical Issue, Support Request, General).

Analytics Dashboard

Provides email statistics:

Total emails

Emails in last 24 hours

Count of urgent emails

Sentiment distribution (Positive, Negative, Neutral)

Visual charts:

Sentiment distribution

Priority breakdown

Extensible analytics module for additional insights.

Architecture & Approach

The system follows a modular design:

app.py                  -> Streamlit UI
src/
 ├─ preprocess.py       -> Email preprocessing & feature extraction
 ├─ reply_generator.py  -> Generates draft replies using RAG
 ├─ rag.py              -> Retrieval-Augmented Generation module
 ├─ analytics.py        -> Dashboard analytics
 ├─ email_retriever.py  -> (Optional) Email fetching module
 └─ db_helper.py        -> SQLite database support
data/
 ├─ intern_emails.csv   -> Sample dataset
 ├─ emails.db           -> SQLite storage
 └─ knowledge_base.csv  -> Contextual KB for RAG


Workflow:

Load Emails: From CSV or database.

Preprocess: Normalize headers, classify priority & sentiment, extract details.

Filter Support Emails: Keep only relevant messages.

RAG-based Reply: Retrieve KB context and generate AI responses.

Display Dashboard: Show inbox and analytics charts.

User Interaction: View emails, edit drafts, and monitor analytics.

Setup & Installation

Clone repository

git clone https://github.com/YOUR_GITHUB_USERNAME/ai-communication-assistant.git
cd ai-communication-assistant


Install dependencies

pip install -r requirements.txt


Set API Key for Groq LLM in secrets.toml:

[GROQ_API_KEY]
api_key = "YOUR_GROQ_API_KEY"

Run the app
streamlit run app.py
Notes

Currently, the app uses CSV and optional SQLite storage for email data.

Email retrieval from external services (Gmail, Outlook) can be added via email_retriever.py.

RAG reply generation requires Groq API key.

Phone numbers and alternate emails are detected only if present in the email body.

Author
Dinesh Narsing Reddy –  Artificial Intelligence and Data Science Student ,Developer & AI/ML Engineer 
LinkedIn: www.linkedin.com/in/dinesh-reddy-narsing-918b23255
