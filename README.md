# Mailbox Knowledge Gap Analyzer

**A Smart Technology knowledge-intelligence solution.**  
*Clear insight. Better service.*

A privacy-first Streamlit app that turns a mailbox export into a ranked,
evidence-backed backlog of help articles. The free analysis works immediately;
an optional paid verification pass discovers topics and separates generic
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

No secret is required for the free workflow. The paid verification tiers run
on the operator's Anthropic key, configured in Streamlit secrets along with
the Stripe payment links, unlock codes and the Formspree lead endpoint:

```
ANTHROPIC_API_KEY = "sk-ant-..."
STRIPE_LINK_49 = "https://buy.stripe.com/..."
STRIPE_LINK_99 = "https://buy.stripe.com/..."
UNLOCK_CODE_49 = "ST-XXXX"
UNLOCK_CODE_99 = "ST-YYYY"
FORMSPREE_ENDPOINT = "https://formspree.io/f/..."
```

Missing secrets degrade gracefully: the free scan, industry packs, sample-data
demo and lead capture all work with no secrets at all.

## Accepted input

`.csv`, `.xlsx`, and `.xls` mailbox exports. Subject and body columns are
auto-detected; sender is optional. Any export with a subject and a
message/body column works; sender is optional.

## Run the tests

The analysis core is fully unit-tested without Streamlit or any API access:

```
python -m unittest discover -s tests -v
```

The original Smart Technology visual identity is documented in `BRAND.md` and
the reusable vector mark is stored in `assets/smart-technology-mark.svg`.
