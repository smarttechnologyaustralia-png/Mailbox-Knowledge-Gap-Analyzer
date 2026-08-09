# Mailbox Knowledge Gap Analyzer

A privacy-first Streamlit app that turns a mailbox export into a ranked,
evidence-backed backlog of help articles. The free analysis works immediately;
an optional Anthropic-powered pass discovers topics and separates generic
questions from case-specific requests. The semantic workflow also produces a
full assessment report and copy-ready first drafts for up to three priority
knowledge articles, grounded in redacted mailbox evidence.

## Run locally

1. Install Python 3.11 or 3.12.
2. Create and activate a virtual environment.
3. Run `pip install -r requirements.txt`.
4. Run `streamlit run app.py`.

The deployment installs a compact spaCy model for Presidio name detection. If
that model is ever unavailable, the app still degrades gracefully to local
email-address and phone-number redaction instead of crashing.

## Share a live link

Push these files to a GitHub repository, then open
[Streamlit Community Cloud](https://share.streamlit.io), choose **Create app**,
select the repository and branch, and set the entry point to `app.py`.
Community Cloud produces a public `*.streamlit.app` link your team can test.

No secret is required for the free workflow. Each tester can enter their own
Anthropic API key inside the optional AI upgrade; the key stays in that
browser session and is not committed to the repository.

## Accepted input

`.csv`, `.xlsx`, and `.xls` mailbox exports. Subject and body columns are
auto-detected; sender is optional. The included
`Sample_IT_Helpdesk_Mailbox_500.xlsx` is ready for a demo.
