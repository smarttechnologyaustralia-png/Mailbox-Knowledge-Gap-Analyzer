"""
Mailbox Knowledge Gap Analyzer — Streamlit demo app (v3)
-----------------------------------------------------------
Upload -> Clean -> Analyze (smart defaults) -> Results (refine & re-run).
Column mapping and topic configuration are both hidden by default and
only surfaced when genuinely needed.

RUN:
    pip install streamlit pandas openpyxl presidio-analyzer presidio-anonymizer
    python -m spacy download en_core_web_lg
    streamlit run app.py
"""
import html
import re
import time

import pandas as pd
import streamlit as st

try:
    import anthropic
    from llm_classifier import discover_topics_with_llm, classify_batch_with_llm, estimate_cost, BATCH_SIZE
    from scoping_logic import quick_keyword_prescan, estimate_time_and_cost, format_time, build_stratified_sample
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

LLM_MODELS = {
    "Fast & cheap (recommended)": "claude-haiku-4-5-20251001",
    "More capable, higher cost": "claude-sonnet-5",
}

st.set_page_config(page_title="Mailbox Knowledge Gap Analyzer", layout="wide", page_icon="📬")

st.markdown("""
<style>
.stApp { background: radial-gradient(circle at 90% 0%, #eef2ff 0, transparent 30%), #f8fafc; }
.block-container { max-width: 1180px; padding-top: 2rem; padding-bottom: 4rem; }
h1, h2, h3 { letter-spacing: -0.025em; color: #172554; }
[data-testid="stFileUploaderDropzone"] { background: white; border: 1.5px dashed #94a3b8; border-radius: 16px; }
[data-testid="stMetric"] { background: white; border: 1px solid #e2e8f0; padding: 16px; border-radius: 14px; }
.hero-kicker { color:#4f46e5; font-size:12px; font-weight:800; letter-spacing:.12em; text-transform:uppercase; }
.step-pill {
    display: inline-block; padding: 6px 16px; border-radius: 20px;
    font-size: 13px; font-weight: 600; margin-right: 8px;
}
.step-done { background: #DCFCE7; color: #166534; }
.step-active { background: #1F3864; color: white; }
.step-pending { background: #E5E7EB; color: #6B7280; }
.article-card {
    background: white; border: 1px solid #E5E7EB; border-radius: 12px;
    padding: 20px 24px; margin-bottom: 16px;
}
.rank-badge {
    display: inline-block; width: 36px; height: 36px; border-radius: 50%;
    text-align: center; line-height: 36px; font-weight: 700; font-size: 15px;
    margin-right: 12px;
}
.badge-high { background: #DCFCE7; color: #166534; }
.badge-medium { background: #FEF3C7; color: #92400E; }
.badge-low { background: #FEE2E2; color: #991B1B; }
.evidence-box {
    background: #F8FAFC; border-left: 3px solid #94A3B8; padding: 10px 14px;
    font-size: 13px; color: #475569; margin: 6px 0; border-radius: 4px;
}
.privacy-note { background:#eef2ff; color:#3730a3; padding:12px 16px; border-radius:10px; font-size:13px; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Core logic
# ============================================================
DEFAULT_TOPICS = {
    "Password Reset": r"password|locked out|reset",
    "VPN / Remote Access": r"\bvpn\b|remote access|certificate",
    "Software Install / Licensing": r"licen[cs]e|install|software",
    "Printer / Hardware": r"printer|mouse|keyboard|laptop|monitor|hardware",
    "New Starter / Access Provisioning": r"new starter|new hire|access request|offboard|provisioning",
    "Email / Outlook Issues": r"outlook|mailbox|calendar|junk folder|spam",
    "WiFi / Network": r"wi-?fi|network|internet connection",
    "Meeting Room / AV": r"conference room|boardroom|\bav\b|screen",
    "Security / Phishing": r"phishing|suspicious email|security policy",
}

QUOTE_MARKERS = re.compile(
    r"(From:\s|Sent:\s|-{3,}\s*Original Message\s*-{3,}|On .{5,80}wrote:|"
    r"________________________________)", re.I)
QUESTION_PATTERN = re.compile(
    r"\?|how do i|how does|what is|what does|where do i|where can i|when is|when do i|"
    r"who is|who do i|can you|could you|please advise|please confirm|not sure|unsure|"
    r"clarif|does this mean|do i need|am i required|is it possible|what happens if|"
    r"could i|can i", re.I)
ACK_PATTERN = re.compile(
    r"^(thanks|thank you|noted|no problem|great|perfect|all good|done|sounds good)\b", re.I)
AUTOMATED_SENDER_MARKERS = ["no-reply", "noreply", "system", "automated", "notification", "service desk"]


def strip_html(text):
    """Removes HTML tags and common entities -- many real mailbox exports
    store the body as HTML, and leaving tags in place corrupts every
    downstream step (quote detection, question detection, topic matching)."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;|&amp;|&lt;|&gt;|&#\d+;", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def strip_quotes(body_text):
    """Removes quoted reply history, keeping only the new message.
    Falls back safely if the quote appears BEFORE the new text instead of
    after it (common outside Outlook's top-posting convention) -- without
    this check, that ordering would silently delete the real message,
    which was found and fixed after testing against a non-Outlook sample."""
    match = QUOTE_MARKERS.search(body_text)
    if not match:
        return body_text
    before = body_text[:match.start()].strip()
    after = body_text[match.end():].strip()
    if len(before) >= 10:
        return before
    if len(after) >= 10:
        return after
    return body_text


def classify_email_type(new_content, sender_name):
    text = new_content.strip()
    sender_lc = (sender_name or "").lower()
    if any(m in sender_lc for m in AUTOMATED_SENDER_MARKERS):
        return "automated"
    if not text:
        return "unclear"
    if ACK_PATTERN.match(text) and len(text) < 120:
        return "acknowledgment"
    if QUESTION_PATTERN.search(text):
        return "genuine_question"
    return "status_update"


def tag_topics(text, topic_patterns):
    text_lc = text.lower()
    return [name for name, pattern in topic_patterns.items() if re.search(pattern, text_lc, re.I)]


@st.cache_resource
def get_presidio_analyzer():
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    configuration = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
    }
    nlp_engine = NlpEngineProvider(nlp_configuration=configuration).create_engine()
    return AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])


def redact_dataframe(df, text_columns, progress_callback=None):
    try:
        analyzer = get_presidio_analyzer()
    except Exception:
        analyzer = None
    pseudonym_map = {}
    counters = {}

    def get_pseudonym(original_text, entity_type):
        key = (entity_type, original_text.strip().lower())
        if key not in pseudonym_map:
            counters[entity_type] = counters.get(entity_type, 0) + 1
            pseudonym_map[key] = f"[{entity_type}_{counters[entity_type]}]"
        return pseudonym_map[key]

    def redact_text(text):
        if not text:
            return text
        if analyzer is None:
            # Safe deployment fallback when Presidio's NLP model is unavailable.
            matches = []
            patterns = {
                "EMAIL_ADDRESS": r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
                "PHONE_NUMBER": r"(?<!\w)(?:\+?\d[\d ()-]{7,}\d)(?!\w)",
            }
            for entity_type, pattern in patterns.items():
                for match in re.finditer(pattern, text, re.I):
                    matches.append((match.start(), match.end(), entity_type))
            results = sorted(matches, key=lambda r: r[0], reverse=True)
            out = text
            for start, end, entity_type in results:
                out = out[:start] + get_pseudonym(text[start:end], entity_type) + out[end:]
            return out
        results = analyzer.analyze(text=text, entities=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"], language="en")
        results = sorted(results, key=lambda r: r.start, reverse=True)
        out = text
        for r in results:
            placeholder = get_pseudonym(text[r.start:r.end], r.entity_type)
            out = out[:r.start] + placeholder + out[r.end:]
        return out

    total = len(df) * len(text_columns)
    done = 0
    for col in text_columns:
        redacted_col = []
        for val in df[col]:
            redacted_col.append(redact_text(str(val)))
            done += 1
            if progress_callback and done % 5 == 0:
                progress_callback(done / total)
        df[col + "_redacted"] = redacted_col
    if progress_callback:
        progress_callback(1.0)
    return df, pseudonym_map


def run_analysis(df, topic_patterns, subject_col, body_col, sender_col):
    df = df.copy()
    df["new_content"] = df[body_col].fillna("").astype(str).apply(strip_html).apply(strip_quotes)
    if not sender_col or sender_col not in df.columns:
        sender_values = pd.Series("", index=df.index)
    else:
        sender_values = df[sender_col].fillna("").astype(str)
    df["email_type"] = [classify_email_type(content, sender) for content, sender in zip(df["new_content"], sender_values)]
    df["topics"] = (df[subject_col].fillna("").astype(str) + " " + df["new_content"]).apply(
        lambda t: tag_topics(t, topic_patterns))

    total = len(df)
    rows = []
    for topic in topic_patterns:
        topic_mask = df["topics"].apply(lambda t: topic in t)
        topic_total = int(topic_mask.sum())
        if topic_total == 0:
            continue
        topic_genuine_mask = topic_mask & (df["email_type"] == "genuine_question")
        topic_genuine = int(topic_genuine_mask.sum())
        pct_genuine = round(100 * topic_genuine / topic_total, 1) if topic_total else 0
        examples = df.loc[topic_genuine_mask, "new_content"].str.strip()
        examples = [e for e in examples if 15 < len(e) < 220][:3]
        rows.append(dict(
            topic=topic, total_emails=topic_total,
            pct_of_mailbox=round(100 * topic_total / total, 1),
            genuine_questions=topic_genuine, pct_genuine=pct_genuine,
            examples=examples,
        ))

    backlog = pd.DataFrame(rows, columns=["topic", "total_emails", "pct_of_mailbox", "genuine_questions", "pct_genuine", "examples"])
    backlog = backlog.sort_values("genuine_questions", ascending=False).reset_index(drop=True)
    if len(backlog):
        backlog.insert(0, "rank", range(1, len(backlog) + 1))

    type_summary = df["email_type"].value_counts().rename_axis("type").reset_index(name="count")
    type_summary["pct"] = round(100 * type_summary["count"] / total, 1)

    return df, backlog, type_summary


def run_llm_analysis(df, subject_col, body_col, api_key, model, sample_size=40,
                      batch_size=None, progress_callback=None, status_callback=None,
                      known_topics=None):
    """
    The AI-powered path: no pre-defined topic list needed -- discovers
    topics from the data itself, then classifies every email with real
    language understanding rather than keyword matching. This is what
    lets it work on a dataset from any industry, and what lets it
    correctly separate generic questions from case-specific ones (the
    thing the rule-based version got wrong on Printer/Hardware).

    known_topics: pass in topics already discovered during the scoping
    step, to avoid paying for and waiting on a second discovery call.
    """
    batch_size = batch_size or BATCH_SIZE
    client = anthropic.Anthropic(api_key=api_key)

    df = df.copy()
    df[subject_col] = df[subject_col].fillna("").astype(str)
    df[body_col] = df[body_col].fillna("").astype(str).apply(strip_html).apply(strip_quotes)
    emails = [{"subject": r[subject_col], "body": r[body_col]} for _, r in df.iterrows()]

    if known_topics:
        topics = known_topics
    else:
        if status_callback:
            status_callback("Reading a sample to discover this mailbox's actual topics...")
        sample = emails[:min(sample_size, len(emails))]
        topics = discover_topics_with_llm(client, model, sample)


    results = []
    n = len(emails)
    for i in range(0, n, batch_size):
        batch = emails[i:i + batch_size]
        if status_callback:
            status_callback(f"Classifying emails {i+1}-{min(i+batch_size, n)} of {n}...")
        try:
            batch_results = classify_batch_with_llm(client, model, batch, topics)
        except Exception:
            # Don't let one bad batch crash the whole run -- mark it
            # unclear and keep going, same philosophy as the rest of
            # this app's error handling.
            batch_results = [{"type": "unclear", "topic": "Other", "specificity": "not_applicable"}
                              for _ in batch]
        results.extend(batch_results)
        if progress_callback:
            progress_callback(min(1.0, (i + batch_size) / n))

    df["email_type"] = [r.get("type", "unclear") for r in results]
    df["topics"] = [[r.get("topic", "Other")] for r in results]
    df["specificity"] = [r.get("specificity", "not_applicable") for r in results]
    df["new_content"] = df[body_col]

    total = len(df)
    rows = []
    for topic in topics:
        topic_mask = df["topics"].apply(lambda t: topic in t)
        topic_total = int(topic_mask.sum())
        if topic_total == 0:
            continue
        genuine_mask = topic_mask & (df["email_type"] == "genuine_question")
        generic_mask = genuine_mask & (df["specificity"] == "generic")
        topic_genuine = int(genuine_mask.sum())
        topic_generic = int(generic_mask.sum())
        pct_genuine = round(100 * topic_genuine / topic_total, 1) if topic_total else 0
        examples = df.loc[generic_mask, body_col].str.strip()
        examples = [e[:220] for e in examples if 15 < len(e)][:3]
        rows.append(dict(
            topic=topic, total_emails=topic_total,
            pct_of_mailbox=round(100 * topic_total / total, 1),
            genuine_questions=topic_generic,  # ranked by GENERIC questions specifically
            pct_genuine=pct_genuine, examples=examples,
        ))

    backlog = pd.DataFrame(rows, columns=["topic", "total_emails", "pct_of_mailbox", "genuine_questions", "pct_genuine", "examples"])
    backlog = backlog.sort_values("genuine_questions", ascending=False).reset_index(drop=True)
    if len(backlog):
        backlog.insert(0, "rank", range(1, len(backlog) + 1))

    type_summary = df["email_type"].value_counts().rename_axis("type").reset_index(name="count")
    type_summary["pct"] = round(100 * type_summary["count"] / total, 1)

    return df, backlog, type_summary


def confidence_band(pct_genuine):
    if pct_genuine >= 70:
        return "High", "badge-high", "🟢"
    if pct_genuine >= 40:
        return "Medium", "badge-medium", "🟡"
    return "Needs review", "badge-low", "🔴"


def suggest_format(topic):
    heuristics = {
        "Password": "Step-by-step guide", "VPN": "Troubleshooting guide", "Software": "FAQ + request form",
        "Printer": "Troubleshooting guide", "New Starter": "Checklist", "Email": "FAQ",
        "WiFi": "FAQ", "Meeting": "Booking guide", "Security": "FAQ + reporting guide",
    }
    for key, fmt in heuristics.items():
        if key.lower() in topic.lower():
            return fmt
    return "FAQ article"


# ============================================================
# Silent column auto-detection
# ============================================================
def auto_detect_column(columns, must_contain_any):
    """Returns (best_guess, confident) -- confident=True means we found a
    strong match and don't need to bother the user about it."""
    cols_lower = {c: c.lower() for c in columns}
    # Pass 1: exact match on common names
    for target in must_contain_any:
        for c, cl in cols_lower.items():
            if cl == target:
                return c, True
    # Pass 2: contains match
    for target in must_contain_any:
        for c, cl in cols_lower.items():
            if target in cl:
                return c, True
    # No confident match found
    return None, False


def guess_body_by_content(df, exclude_cols):
    """When column names give no clues, the body column is reliably the
    one with the longest average text -- email bodies are almost always
    longer than subjects, senders, or dates."""
    candidates = [c for c in df.columns if c not in exclude_cols]
    if not candidates:
        return df.columns[0]
    avg_lengths = {c: df[c].astype(str).str.len().mean() for c in candidates}
    return max(avg_lengths, key=avg_lengths.get)


def guess_sender_by_content(df, exclude_cols):
    """When column names give no clues, a column where most values contain
    '@' is very likely a sender/email field."""
    candidates = [c for c in df.columns if c not in exclude_cols]
    for c in candidates:
        vals = df[c].astype(str)
        if (vals.str.contains("@")).mean() > 0.5:
            return c
    return None


def detect_all_columns(df):
    cols = df.columns.tolist()
    subject_col, subj_conf = auto_detect_column(cols, ["subject", "email subject"])
    body_col, body_conf = auto_detect_column(cols, ["body", "message", "content"])
    sender_col, sender_conf = auto_detect_column(cols, ["sender_name", "sender name", "from name", "sender", "from"])

    # Content-based fallback when names give literally no signal at all --
    # a smarter starting guess than just grabbing the first column.
    used = set()
    if body_col is None:
        body_col = guess_body_by_content(df, used)
    used.add(body_col)
    if sender_col is None:
        sender_col = guess_sender_by_content(df, used)
    if sender_col is not None:
        used.add(sender_col)
    if subject_col is None:
        remaining = [c for c in cols if c not in used]
        subject_col = remaining[0] if remaining else cols[0]

    # Sender is optional: a valid export may not contain it at all.
    all_confident = subj_conf and body_conf
    return dict(subject=subject_col, body=body_col, sender=sender_col), all_confident


# ============================================================
# Session state
# ============================================================
if "stage" not in st.session_state:
    st.session_state.stage = "upload"
if "topic_patterns" not in st.session_state:
    st.session_state.topic_patterns = dict(DEFAULT_TOPICS)


def step_indicator():
    if st.session_state.get("use_llm"):
        steps = ["1. Upload", "2. Clean", "3. Scope", "4. Analyze", "5. Results"]
        current = {"upload": 0, "cleaning": 1, "scoping": 2, "analyzing": 3, "results": 4}[st.session_state.stage]
    else:
        steps = ["1. Upload", "2. Clean", "3. Analyze", "4. Results"]
        current = {"upload": 0, "cleaning": 1, "analyzing": 2, "results": 3}[st.session_state.stage]
    html = ""
    for i, s in enumerate(steps):
        cls = "step-done" if i < current else ("step-active" if i == current else "step-pending")
        html += f'<span class="step-pill {cls}">{s}</span>'
    st.markdown(html, unsafe_allow_html=True)
    st.write("")


st.markdown('<div class="hero-kicker">Support intelligence · privacy first</div>', unsafe_allow_html=True)
st.title("📬 Mailbox Knowledge Gap Analyzer")
st.caption("Turn repeated questions into a defensible, evidence-backed help-article backlog.")
step_indicator()

# ============================================================
# STAGE 1: Upload — no column dropdowns, no topic config.
# ============================================================
if st.session_state.stage == "upload":
    st.subheader("Upload a mailbox export to begin")
    st.caption("We'll protect personal data first, then find which self-service articles would cut the most repeat emails.")

    c1, c2, c3 = st.columns(3)
    c1.info("🔒 **Privacy-first**\n\nNames, emails, and phone numbers are removed before any analysis runs.")
    c2.info("🎯 **Evidence-based**\n\nEvery recommendation is backed by real examples, not a black-box score.")
    c3.info("📊 **Ranked output**\n\nSee exactly which article to write first, and why.")

    st.write("")
    uploaded_file = st.file_uploader("Mailbox export (.xlsx, .xls, or .csv)", type=["xlsx", "xls", "csv"])

    if uploaded_file:
        try:
            if uploaded_file.name.lower().endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Couldn't read this file: {e}. Check it's a valid, unlocked .xlsx/.xls/.csv export.")
            st.stop()

        if len(df) == 0:
            st.error("This file has no rows to analyze.")
            st.stop()

        # Safety valve for very large files in a live setting -- keeps
        # redaction (the slow step) from taking minutes in front of an
        # audience. Full analysis is still possible, just not the default.
        MAX_ROWS_DEFAULT = 1000
        if len(df) > MAX_ROWS_DEFAULT:
            st.warning(f"This file has {len(df)} rows. For a live demo, analyzing a sample of "
                       f"the first {MAX_ROWS_DEFAULT} is much faster.")
            use_sample = st.radio("How much to analyze?",
                                   [f"Sample the first {MAX_ROWS_DEFAULT} rows (fast, good for a demo)",
                                    f"Analyze all {len(df)} rows (accurate, may take a while)"])
            if use_sample.startswith("Sample"):
                df = df.sample(n=MAX_ROWS_DEFAULT, random_state=42).sort_index().copy()
        st.session_state.raw_df = df

        detected, confident = detect_all_columns(df)
        st.session_state.subject_col = detected["subject"]
        st.session_state.body_col = detected["body"]
        st.session_state.sender_col = detected["sender"]

        st.success(f"Loaded **{len(df)} emails** and identified the subject, message, and sender fields automatically.")

        if not confident:
            st.warning("Couldn't confidently identify all fields — please confirm below.")
            cols = df.columns.tolist()
            c1, c2, c3 = st.columns(3)
            with c1:
                st.session_state.subject_col = st.selectbox("Subject column", cols, index=cols.index(detected["subject"]))
            with c2:
                st.session_state.body_col = st.selectbox("Body column", cols, index=cols.index(detected["body"]))
            with c3:
                sender_options = ["(No sender column)"] + cols
                sender_default = detected["sender"] or "(No sender column)"
                sender_choice = st.selectbox("Sender name column (optional)", sender_options, index=sender_options.index(sender_default))
                st.session_state.sender_col = None if sender_choice == "(No sender column)" else sender_choice
        else:
            with st.expander("⚙️ Advanced: change detected columns"):
                cols = df.columns.tolist()
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.session_state.subject_col = st.selectbox("Subject column", cols, index=cols.index(st.session_state.subject_col))
                with c2:
                    st.session_state.body_col = st.selectbox("Body column", cols, index=cols.index(st.session_state.body_col))
                with c3:
                    sender_options = ["(No sender column)"] + cols
                    sender_default = st.session_state.sender_col or "(No sender column)"
                    sender_choice = st.selectbox("Sender name column (optional)", sender_options, index=sender_options.index(sender_default))
                    st.session_state.sender_col = None if sender_choice == "(No sender column)" else sender_choice

        st.write("")
        if st.button("Start →", type="primary"):
            st.session_state.use_llm = False  # always starts with the free, automatic pass
            st.session_state.stage = "cleaning"
            st.rerun()

# ============================================================
# STAGE 2: Cleaning
# ============================================================
elif st.session_state.stage == "cleaning":
    st.subheader("Step 2 — Protecting personal information")
    df = st.session_state.raw_df

    st.write("Before any analysis runs, every name, email address, and phone number is found and replaced with a placeholder — locally, before anything is examined further.")

    progress = st.progress(0.0)
    status = st.empty()
    if "working_df" not in st.session_state:
        status.text("Removing HTML and quoted reply history...")
        prepared = df.copy()
        body_col = st.session_state.body_col
        prepared[body_col] = prepared[body_col].fillna("").astype(str).apply(strip_html).apply(strip_quotes)
        prepared[st.session_state.subject_col] = prepared[st.session_state.subject_col].fillna("").astype(str).apply(strip_html)
        progress.progress(0.15)
        working_df, pseudonym_map = redact_dataframe(
            prepared, [st.session_state.subject_col, body_col],
            progress_callback=lambda p: (progress.progress(0.15 + p * 0.85), status.text(f"Redacting personal details... {int(p*100)}%")))
        st.session_state.working_df = working_df
        st.session_state.pseudonym_count = len(pseudonym_map)
    working_df = st.session_state.working_df
    status.empty()
    progress.progress(1.0)
    st.success(f"✅ **{len(pseudonym_map)} unique names, emails, and phone numbers protected.** Nothing identifiable leaves this step.")

    example_row = df.iloc[0]
    example_redacted = working_df.iloc[0][st.session_state.body_col + "_redacted"]
    with st.expander("See an example of what changed"):
        c1, c2 = st.columns(2)
        c1.markdown("**Before**")
        c1.markdown(f'<div class="evidence-box">{html.escape(str(example_row[st.session_state.body_col])[:200])}</div>', unsafe_allow_html=True)
        c2.markdown("**After**")
        c2.markdown(f'<div class="evidence-box">{html.escape(str(example_redacted)[:200])}</div>', unsafe_allow_html=True)

    if st.button("Continue →", type="primary"):
        st.session_state.stage = "scoping" if st.session_state.get("use_llm") else "analyzing"
        st.rerun()

# ============================================================
# STAGE 2.5: Scoping — only for AI-powered mode. Shows real cost/time
# for the full dataset, and lets the user choose a smaller, smartly
# stratified sample instead, with live-updating cost/time as they adjust.
# ============================================================
elif st.session_state.stage == "scoping":
    st.subheader("Choose how much to analyze")
    working_df = st.session_state.working_df
    subj_col = st.session_state.subject_col + "_redacted"
    body_col = st.session_state.body_col + "_redacted"
    total_n = len(working_df)

    if "scoping_topics" not in st.session_state:
        with st.spinner("Reading a small sample to see what topics are actually in this mailbox..."):
            client = anthropic.Anthropic(api_key=st.session_state.api_key)
            sample_emails = [
                {"subject": r[subj_col], "body": r[body_col]}
                for _, r in working_df.head(40).iterrows()
            ]
            try:
                st.session_state.scoping_topics = discover_topics_with_llm(
                    client, st.session_state.llm_model, sample_emails)
            except Exception as e:
                st.error(f"Couldn't reach the API to preview topics: {e}")
                st.stop()

    topics = st.session_state.scoping_topics

    with st.spinner("Estimating volume per topic (free, no API calls)..."):
        working_df["_rough_topic"] = quick_keyword_prescan(working_df, subj_col, body_col, topics)

    st.success(f"Found **{len(topics)} likely topics** in this mailbox: {', '.join(topics)}")

    rough_counts = working_df["_rough_topic"].value_counts()
    st.bar_chart(rough_counts)
    st.caption("This breakdown is a free, instant keyword estimate — not the final AI classification. It's only here to size the cost/time choice below.")

    st.write("")
    full_cost, full_secs = estimate_time_and_cost(total_n, BATCH_SIZE, st.session_state.llm_model)

    st.markdown(f"**Full dataset: {total_n} emails** — estimated cost **${full_cost}**, estimated time **{format_time(full_secs)}**")

    choice = st.radio(
        "How do you want to proceed?",
        [f"Analyze all {total_n} emails (most accurate, takes longest)",
         "Analyze a smaller, representative sample (faster, cheaper — every topic still gets fair coverage)"],
    )

    if choice.startswith("Analyze a smaller"):
        default_sample = min(total_n, max(100, int(total_n * 0.15)))
        slider_min = min(total_n, max(1, len(topics) * 5))
        default_sample = max(slider_min, default_sample)
        sample_size = st.slider("Sample size", min_value=slider_min, max_value=total_n,
                                 value=default_sample, step=1 if total_n < 100 else 10)
        sample_cost, sample_secs = estimate_time_and_cost(sample_size, BATCH_SIZE, st.session_state.llm_model)
        st.markdown(f"**Selected: {sample_size} emails** — estimated cost **${sample_cost}**, estimated time **{format_time(sample_secs)}**")

        preview_sample = build_stratified_sample(working_df, "_rough_topic", sample_size, min_per_topic=8)
        with st.expander("See how this sample would be split across topics"):
            st.dataframe(preview_sample["_rough_topic"].value_counts().rename_axis("topic").reset_index(name="sample_count"),
                         use_container_width=True, hide_index=True)

        st.session_state.chosen_scope = "sample"
        st.session_state.chosen_sample_size = sample_size
    else:
        st.session_state.chosen_scope = "full"

    st.write("")
    if st.button("Confirm and run analysis →", type="primary"):
        st.session_state.stage = "analyzing"
        st.rerun()

# ============================================================
# STAGE 3: Analyzing — uses smart defaults automatically, or the AI path.
# ============================================================
elif st.session_state.stage == "analyzing":
    st.subheader("Step 3 — Finding patterns")
    status = st.empty()
    progress = st.progress(0.0)

    if st.session_state.get("use_llm"):
        # Use whichever scope was chosen in the Scoping step: the full
        # working dataset, or the smaller stratified sample.
        if st.session_state.get("chosen_scope") == "sample":
            data_to_analyze = build_stratified_sample(
                st.session_state.working_df, "_rough_topic",
                st.session_state.chosen_sample_size, min_per_topic=8)
            st.caption(f"Analyzing a stratified sample of {len(data_to_analyze)} emails "
                       f"(out of {len(st.session_state.working_df)} total) — chosen in the previous step.")
        else:
            data_to_analyze = st.session_state.working_df
            st.caption(f"Analyzing all {len(data_to_analyze)} emails, as chosen in the previous step.")

        try:
            analyzed_df, backlog, type_summary = run_llm_analysis(
                data_to_analyze,
                st.session_state.subject_col + "_redacted", st.session_state.body_col + "_redacted",
                st.session_state.api_key, st.session_state.llm_model,
                progress_callback=progress.progress, status_callback=status.text,
                known_topics=st.session_state.get("scoping_topics"),
            )
            st.session_state.sample_size_used = len(data_to_analyze)
            st.session_state.population_size = len(st.session_state.working_df)
            status.empty()
            st.success("✅ Analysis complete — topics were discovered directly from this mailbox, not from a pre-set list.")
        except Exception as e:
            status.empty()
            st.error(f"AI-powered analysis failed: {e}")
            st.caption("Check your API key is valid and has available credit, or switch back to fast/rule-based mode by starting over.")
            st.stop()
    else:
        stages = [
            (0.25, "Separating new messages from quoted reply history..."),
            (0.50, "Classifying each email — question, status update, or acknowledgment..."),
            (0.75, "Grouping into topics using a general-purpose starting taxonomy..."),
            (1.00, "Ranking by genuine, answerable question volume..."),
        ]
        for pct, msg in stages:
            status.text(msg)
            progress.progress(pct)
            time.sleep(0.5)

        analyzed_df, backlog, type_summary = run_analysis(
            st.session_state.working_df, st.session_state.topic_patterns,
            st.session_state.subject_col + "_redacted", st.session_state.body_col + "_redacted",
            st.session_state.sender_col)
        status.empty()
        st.success("✅ Analysis complete, using a general-purpose starting taxonomy. You can refine the topic categories from the results screen.")

    result_key = "ai" if st.session_state.get("use_llm") else "free"
    st.session_state[f"{result_key}_analyzed_df"] = analyzed_df
    st.session_state[f"{result_key}_backlog"] = backlog
    st.session_state[f"{result_key}_type_summary"] = type_summary
    st.session_state.active_result = result_key

    st.session_state.stage = "results"
    st.rerun()

# ============================================================
# STAGE 4: Results — topic refinement lives here now.
# ============================================================
elif st.session_state.stage == "results":
    has_ai = "ai_backlog" in st.session_state
    if has_ai:
        view_choice = st.radio(
            "Which results to view?",
            ["🧠 AI-powered (more accurate)", "⚡ Fast/keyword-based"],
            horizontal=True,
            index=0 if st.session_state.get("active_result") == "ai" else 1,
        )
        active = "ai" if view_choice.startswith("🧠") else "free"
    else:
        active = "free"

    analyzed_df = st.session_state[f"{active}_analyzed_df"]
    backlog = st.session_state[f"{active}_backlog"]
    type_summary = st.session_state[f"{active}_type_summary"]
    total = len(analyzed_df)
    genuine_total = int((analyzed_df["email_type"] == "genuine_question").sum())
    opportunity_score = round(100 * genuine_total / total)

    st.subheader("Step 4 — Results")

    if active == "ai" and st.session_state.get("chosen_scope") == "sample":
        pop = st.session_state.get("population_size", total)
        used = st.session_state.get("sample_size_used", total)
        st.info(f"📐 **These results are based on a sample of {used} emails, "
                f"out of {pop} total in this mailbox** — not every email was individually "
                f"classified. Topic percentages below are from the sample directly; treat "
                f"them as an estimate of the full mailbox, not an exact count.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Emails analyzed", total)
    c2.metric("Genuine questions found", genuine_total, f"{100*genuine_total/total:.0f}% of mailbox")
    c3.metric("Topics identified", len(backlog))
    c4.metric("Names/emails/phones protected", st.session_state.pseudonym_count)

    st.write("")
    st.markdown(f"### 🎯 Self-Service Opportunity Score: **{opportunity_score}/100**")
    st.caption("Share of this mailbox that looks like a genuine, answerable question — the honest ceiling on what self-service content could address. Not a promise of how many emails would actually stop.")
    st.progress(opportunity_score / 100)

    if "celebrated" not in st.session_state:
        st.balloons()
        st.session_state.celebrated = True

    st.write("")
    st.markdown("### What's actually in this mailbox")
    st.bar_chart(type_summary.set_index("type")["count"])

    st.write("")
    st.markdown("### 📚 Ranked knowledge article backlog")
    st.caption("Ranked by genuine, answerable question volume per topic — not raw email count.")

    if len(backlog) == 0:
        st.warning(
            "**No topics matched this mailbox.** The starting topic list is tuned for an IT "
            "Helpdesk mailbox (password resets, VPN, etc.) — if this is a different kind of "
            "mailbox, open **'🔧 Refine topic categories & re-run'** below and replace the "
            "topic list with keywords that match what this mailbox is actually about, then "
            "re-run the analysis."
        )

    medals = {1: "🥇", 2: "🥈", 3: "🥉"}
    for _, row in backlog.iterrows():
        label, cls, dot = confidence_band(row["pct_genuine"])
        medal = medals.get(row["rank"], f'{row["rank"]}')
        fmt = suggest_format(row["topic"])

        st.markdown(f"""
        <div class="article-card">
            <span style="font-size:22px;">{medal}</span>
            <span style="font-size:18px; font-weight:700;"> {row['topic']}</span>
            <span class="rank-badge {cls}" style="width:auto; border-radius:20px; padding:4px 12px; margin-left:10px;">{dot} {label} confidence</span>
            <br><br>
            <b>{row['genuine_questions']}</b> genuine questions out of <b>{row['total_emails']}</b> emails on this topic
            ({row['pct_genuine']}% of this topic's volume) &nbsp;·&nbsp; suggested format: <b>{fmt}</b>
        </div>
        """, unsafe_allow_html=True)

        with st.expander(f"See supporting evidence for '{row['topic']}'"):
            if row["examples"]:
                for ex in row["examples"]:
                    st.markdown(f'<div class="evidence-box">{html.escape(str(ex))}</div>', unsafe_allow_html=True)
            else:
                st.caption("No clean example snippets available for this topic.")

    st.write("")
    st.download_button(
        "⬇ Download full backlog as CSV",
        backlog.drop(columns=["examples"]).to_csv(index=False).encode("utf-8"),
        "knowledge_backlog.csv", "text/csv",
    )

    # ---- Refinement, now living here instead of pre-analysis ----
    # Only applies to rule-based mode -- AI-powered mode discovers topics
    # from the data itself rather than using editable keyword rules.
    st.write("")
    st.divider()
    if active == "free":
        with st.expander("🔧 Refine topic categories & re-run analysis"):
            st.caption("Adjust the keyword rules below, add a topic that's missing, or remove one that doesn't apply — then re-run using the same cleaned data (no need to re-upload or re-redact).")
            topics_df = pd.DataFrame([{"topic": k, "keyword_pattern": v} for k, v in st.session_state.topic_patterns.items()])
            edited = st.data_editor(topics_df, num_rows="dynamic", use_container_width=True, key="topic_editor")
            if st.button("Re-analyze with these topics"):
                st.session_state.topic_patterns = {r["topic"]: r["keyword_pattern"] for _, r in edited.iterrows()
                                                    if r["topic"] and r["keyword_pattern"]}
                st.session_state.use_llm = False
                st.session_state.stage = "analyzing"
                st.session_state.pop("celebrated", None)
                st.rerun()
    else:
        st.caption("Topics were discovered automatically from this mailbox's actual content — there's no fixed rule list to edit in AI-powered mode.")

    if not has_ai:
        st.write("")
        st.divider()
        st.markdown("### 🧠 Want more accurate results?")
        st.caption(
            "The results above use keyword matching, which works well for familiar topics but "
            "can misjudge things like generic vs. case-specific questions. AI-powered analysis "
            "reads each email properly, works on any industry or language, and doesn't need a "
            "pre-set topic list — you'll see the real cost and time before anything runs."
        )
        if not LLM_AVAILABLE:
            st.info("Install `anthropic` (`pip install anthropic`) to unlock this.")
        else:
            with st.expander("Set up AI-powered analysis"):
                c1, c2 = st.columns(2)
                with c1:
                    api_key_input = st.text_input("Anthropic API key", type="password",
                                                   help="Never stored beyond this session.")
                with c2:
                    model_choice = st.selectbox("Model", list(LLM_MODELS.keys()))
                if st.button("Continue →", type="primary", disabled=not api_key_input):
                    st.session_state.api_key = api_key_input
                    st.session_state.llm_model = LLM_MODELS[model_choice]
                    st.session_state.use_llm = True
                    st.session_state.pop("scoping_topics", None)  # force fresh discovery
                    st.session_state.stage = "scoping"
                    st.rerun()
                if not api_key_input:
                    st.caption("Enter an API key to continue.")

    st.write("")
    if st.button("↺ Start over with a new file"):
        for key in list(st.session_state.keys()):
            st.session_state.pop(key, None)
        st.rerun()
