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
    from llm_classifier import discover_topics_with_llm, classify_batch_with_llm, draft_kb_articles_with_llm, estimate_cost, BATCH_SIZE
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
.stApp { background: #f6f8fc; color: #15233c; }
[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1120px; padding-top: 3.25rem; padding-bottom: 5rem; }
h1, h2, h3 { letter-spacing: -0.035em; color: #112344; }
h1 a, h2 a, h3 a { display: none !important; }
p { line-height: 1.65; }
.hero {
    position: relative; overflow: hidden; color: white;
    background: linear-gradient(120deg, #0b1f3a 0%, #183d65 58%, #176b78 100%);
    border-radius: 24px; padding: 34px 38px 32px; margin: 0 0 22px;
    box-shadow: 0 20px 60px rgba(15, 35, 64, .14);
}
.hero:after { content:""; position:absolute; width:260px; height:260px; border-radius:50%; right:-70px; top:-120px; background:rgba(255,255,255,.08); }
.hero:before { content:""; position:absolute; width:170px; height:170px; border-radius:50%; right:150px; bottom:-135px; background:rgba(142,227,220,.12); }
.hero-kicker { color:#8ee3dc; font-size:11px; font-weight:800; letter-spacing:.16em; text-transform:uppercase; margin-bottom:9px; }
.hero-title { font-size:clamp(28px,4vw,44px); line-height:1.08; letter-spacing:-.045em; font-weight:780; margin:0 0 10px; }
.hero-copy { color:#cfdae8; font-size:15px; max-width:690px; margin:0; }
.hero-chip { display:inline-flex; align-items:center; gap:7px; margin-top:20px; padding-top:14px; border-top:1px solid rgba(255,255,255,.14); color:#c9dce3; font-size:11px; font-weight:650; letter-spacing:.02em; }
.steps { display:flex; align-items:center; gap:8px; margin:18px 0 42px; flex-wrap:wrap; }
.step-pill {
    display: inline-flex; align-items:center; padding: 8px 14px; border-radius: 999px;
    font-size: 12px; font-weight: 750; border:1px solid transparent;
}
.step-done { background: #e3f8f3; color: #126b5e; border-color:#c7eee5; }
.step-done:before { content:'✓'; display:inline-grid; place-items:center; width:18px; height:18px; margin-right:6px; border-radius:50%; color:white; background:#20a38f; font-size:11px; }
.step-active { background: #16385f; color: white; box-shadow:0 7px 18px rgba(22,56,95,.18); }
.step-pending { background: #edf1f6; color: #768399; border-color:#e1e6ed; }
.feature-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin:22px 0 26px; }
.feature-card { background:#fff; border:1px solid #e4e9f0; border-radius:16px; padding:20px; box-shadow:0 5px 20px rgba(17,35,68,.04); transition:transform .2s ease, box-shadow .2s ease, border-color .2s ease; }
.feature-card:hover { transform:translateY(-3px); border-color:#b9ddd8; box-shadow:0 12px 30px rgba(17,35,68,.09); }
.feature-icon { width:38px; height:38px; display:flex; align-items:center; justify-content:center; border-radius:9px; background:#e8f7f5; color:#126b5e; font-size:11px; font-weight:850; letter-spacing:.06em; margin-bottom:14px; }
.feature-title { font-weight:750; color:#152c4f; font-size:14px; margin-bottom:5px; }
.feature-copy { color:#64748b; font-size:13px; line-height:1.5; }
[data-testid="stFileUploader"] { background:white; padding:22px; border:1px solid #e4e9f0; border-radius:18px; box-shadow:0 8px 25px rgba(17,35,68,.045); }
[data-testid="stFileUploaderDropzone"] { background:#f8fbfd; border:1.5px dashed #91a7b9; border-radius:13px; }
[data-testid="stMetric"] { background: white; border: 1px solid #e4e9f0; padding: 18px; border-radius: 16px; box-shadow:0 6px 20px rgba(17,35,68,.04); }
[data-testid="stMetricLabel"] { color:#64748b; }
[data-testid="stMetricValue"] { color:#112344; }
.stButton > button, .stDownloadButton > button { border-radius:10px; min-height:44px; font-weight:700; padding-left:20px; padding-right:20px; }
.stButton > button[kind="primary"] { background:#147d76; border-color:#147d76; box-shadow:0 7px 18px rgba(20,125,118,.18); }
.stButton > button[kind="primary"]:hover { background:#0f6a64; border-color:#0f6a64; }
[data-testid="stExpander"] { background:white; border:1px solid #e4e9f0; border-radius:12px; overflow:hidden; }
[data-testid="stProgress"] > div > div > div { background-color:#16867d; }
.article-card {
    background: white; border: 1px solid #e4e9f0; border-radius: 16px;
    padding: 22px 24px; margin: 14px 0 8px; box-shadow:0 6px 20px rgba(17,35,68,.04);
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
.redaction-token { display:inline-block; background:#d9f1ec; color:#0d6258; border:1px solid #b8e2da; border-radius:4px; padding:0 4px; font-weight:750; }
.privacy-note { background:#e8f7f5; color:#126b5e; padding:12px 16px; border-radius:10px; font-size:13px; }
.mission-card { display:flex; align-items:center; justify-content:space-between; gap:18px; background:#ffffff; border:1px solid #dfe6ed; border-left:4px solid #147d76; border-radius:10px; padding:17px 20px; margin:8px 0 22px; }
.mission-label { color:#147d76; font-size:10px; font-weight:850; letter-spacing:.14em; text-transform:uppercase; }
.mission-title { color:#173456; font-weight:760; font-size:15px; margin-top:3px; }
.xp-pill { flex:none; background:#edf3f7; color:#34506d; padding:8px 12px; border-radius:6px; font-size:10px; font-weight:800; letter-spacing:.06em; text-transform:uppercase; }
.achievement-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin:16px 0 28px; }
.achievement { display:flex; align-items:center; gap:12px; background:white; border:1px solid #e4e9f0; border-radius:14px; padding:14px 16px; }
.achievement-icon { display:grid; place-items:center; width:38px; height:38px; flex:none; border-radius:8px; background:#e8f7f5; color:#126b5e; border:1px solid #cde8e3; font-size:10px; font-weight:850; letter-spacing:.05em; }
.achievement strong { display:block; color:#173456; font-size:12px; }
.achievement span { color:#78869a; font-size:10px; }
.score-panel { display:flex; align-items:center; gap:22px; background:linear-gradient(135deg,#102b4c,#17516a); color:white; border-radius:18px; padding:22px 26px; margin:22px 0 30px; box-shadow:0 14px 35px rgba(16,43,76,.14); }
.score-ring { --score:0; width:92px; height:92px; flex:none; border-radius:50%; display:grid; place-items:center; background:conic-gradient(#63d9c7 calc(var(--score)*1%),rgba(255,255,255,.13) 0); position:relative; }
.score-ring:after { content:""; position:absolute; inset:8px; border-radius:50%; background:#173b59; }
.score-value { position:relative; z-index:1; font-weight:850; font-size:23px; }
.score-copy strong { display:block; color:white; font-size:18px; margin-bottom:5px; }
.score-copy span { color:#c9dce3; font-size:12px; line-height:1.5; }
.rank-one { border-left:4px solid #147d76; background:linear-gradient(120deg,#f6fcfb,#fff); }
.upgrade-panel { position:relative; overflow:hidden; background:linear-gradient(125deg,#0c223d 0%,#153d5d 62%,#11675f 100%); color:white; border-radius:20px; padding:30px 32px; margin:12px 0 18px; box-shadow:0 18px 45px rgba(12,34,61,.16); }
.upgrade-panel:after { content:""; position:absolute; width:240px; height:240px; right:-95px; top:-120px; border-radius:50%; background:rgba(109,224,203,.1); }
.upgrade-eyebrow { color:#78d8ca; font-size:10px; font-weight:850; letter-spacing:.15em; text-transform:uppercase; margin-bottom:9px; }
.upgrade-title { color:white; font-size:27px; line-height:1.18; font-weight:780; letter-spacing:-.035em; max-width:720px; margin-bottom:9px; }
.upgrade-copy { color:#c9d8e3; font-size:13px; line-height:1.65; max-width:780px; }
.upgrade-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:10px; margin-top:22px; }
.upgrade-benefit { background:rgba(255,255,255,.07); border:1px solid rgba(255,255,255,.11); border-radius:11px; padding:14px; }
.upgrade-benefit strong { display:block; color:white; font-size:12px; margin-bottom:5px; }
.upgrade-benefit span { display:block; color:#bcd0dd; font-size:10px; line-height:1.5; }
.comparison { width:100%; border-collapse:separate; border-spacing:0; margin:0 0 18px; overflow:hidden; border:1px solid #dfe6ed; border-radius:12px; background:white; }
.comparison th { background:#f2f5f8; color:#40546c; font-size:10px; letter-spacing:.07em; text-transform:uppercase; padding:11px 14px; text-align:left; }
.comparison td { color:#40546c; font-size:11px; padding:11px 14px; border-top:1px solid #e8edf2; }
.comparison td:last-child { color:#11675f; font-weight:720; }
.commercial-note { background:#eef8f6; border:1px solid #cfe8e3; border-radius:10px; padding:13px 15px; color:#315d58; font-size:11px; line-height:1.55; margin-bottom:14px; }
.report-cta { position:relative; overflow:hidden; display:grid; grid-template-columns:1fr auto; align-items:center; gap:24px; background:linear-gradient(120deg,#f1faf8 0%,#ffffff 55%,#edf4f8 100%); border:1px solid #bcded8; border-left:5px solid #147d76; border-radius:16px; padding:24px 26px; margin:26px 0 12px; box-shadow:0 10px 30px rgba(15,58,74,.08); }
.report-cta:after { content:""; position:absolute; width:150px; height:150px; right:-70px; top:-80px; border-radius:50%; background:rgba(20,125,118,.07); }
.report-cta-label { color:#147d76; font-size:9px; font-weight:900; letter-spacing:.16em; text-transform:uppercase; margin-bottom:7px; }
.report-cta-title { color:#102b4c; font-size:21px; font-weight:800; letter-spacing:-.025em; line-height:1.25; margin-bottom:6px; }
.report-cta-copy { color:#5d6f81; font-size:11px; line-height:1.55; max-width:720px; }
.report-cta-status { position:relative; z-index:1; flex:none; background:#102b4c; color:white; border-radius:9px; padding:11px 14px; font-size:9px; font-weight:850; letter-spacing:.09em; text-transform:uppercase; white-space:nowrap; }
.deliverable-line { display:flex; flex-wrap:wrap; gap:7px; margin-top:12px; }
.deliverable-line span { background:white; color:#315d58; border:1px solid #d7e9e5; border-radius:999px; padding:5px 9px; font-size:9px; font-weight:700; }
@media (prefers-reduced-motion: reduce) { .feature-card { transition:none; } }
@media (max-width: 720px) {
  .block-container { padding-top:1.5rem; }
  .hero { padding:26px 22px; border-radius:18px; }
  .feature-grid { grid-template-columns:1fr; }
  .achievement-grid { grid-template-columns:1fr; }
  .upgrade-grid { grid-template-columns:1fr; }
  .report-cta { grid-template-columns:1fr; }
  .report-cta-status { width:max-content; }
  .score-panel { align-items:flex-start; }
  .steps { margin-bottom:30px; }
}
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
NON_PERSON_TERMS = {
    "wifi", "wi-fi", "vpn", "outlook", "windows", "microsoft", "teams", "zoom",
    "printer", "password", "mailbox", "internet", "software", "hardware", "helpdesk",
    "service desk", "sharepoint", "onedrive", "excel", "word", "powerpoint",
}


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
        # Named-entity models can confidently mistake product or technology
        # names (for example "WiFi") for people. Keep email/phone recognizers,
        # but apply a quality threshold and domain guardrail to PERSON results.
        filtered_results = []
        for result in results:
            candidate = text[result.start:result.end].strip()
            if result.entity_type == "PERSON":
                if result.score < 0.70:
                    continue
                if candidate.lower() in NON_PERSON_TERMS:
                    continue
                if not re.fullmatch(r"[A-Za-z][A-Za-z .'-]{1,79}", candidate):
                    continue
                if candidate.isupper() and len(candidate) <= 6:
                    continue
            filtered_results.append(result)
        results = filtered_results
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
        return "High", "badge-high", "●"
    if pct_genuine >= 40:
        return "Medium", "badge-medium", "●"
    return "Needs review", "badge-low", "●"


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


def validate_mailbox_dataframe(df, detected, confident):
    """Rejects readable spreadsheets that do not contain usable email data.

    Named subject/body fields are strong evidence of a mailbox export. When
    names are unfamiliar and content-based fallback is needed, require a small
    body of message-like text so templates, trackers and mostly empty sheets do
    not pass as mailboxes merely because pandas can read them.
    """
    if df is None or df.empty or len(df.columns) == 0:
        return False, "The file does not contain any data rows."

    body_col = detected.get("body")
    subject_col = detected.get("subject")
    if body_col not in df.columns:
        return False, "A usable message-body field could not be identified."

    def clean_value(value):
        if pd.isna(value):
            return ""
        value = re.sub(r"\s+", " ", str(value)).strip()
        return "" if value.lower() in {"nan", "none", "null"} else value

    body_text = df[body_col].map(clean_value)
    subject_text = df[subject_col].map(clean_value) if subject_col in df.columns else pd.Series("", index=df.index)
    usable_mask = (body_text.str.len() > 0) | (subject_text.str.len() > 0)
    usable_count = int(usable_mask.sum())
    nonempty_body_count = int((body_text.str.len() > 0).sum())

    if usable_count == 0 or nonempty_body_count == 0:
        return False, "The file has no usable email subject or message content."

    if not confident:
        # Unknown column names are allowed, but the fallback body must resemble
        # a collection of messages rather than headings or spreadsheet labels.
        message_like_count = int((body_text.str.len() >= 25).sum())
        substantial_count = int((body_text.str.len() >= 60).sum())
        if message_like_count < 2 and substantial_count < 1:
            return False, (
                "The spreadsheet is readable, but it does not appear to contain mailbox data. "
                "No subject/body columns were recognised and there is not enough message-like text to analyze."
            )

    return True, ""


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
    st.markdown(f'<div class="steps">{html}</div>', unsafe_allow_html=True)


st.markdown("""
<section class="hero">
  <div class="hero-kicker">Support intelligence · privacy first</div>
  <div class="hero-title">Mailbox Knowledge Gap Analyzer</div>
  <p class="hero-copy">Turn recurring support questions into a ranked, evidence-backed content plan—without exposing personal information.</p>
  <div class="hero-chip">FOUR-STAGE REVIEW &nbsp;·&nbsp; PRIVACY-LED &nbsp;·&nbsp; EVIDENCE-BASED</div>
</section>
""", unsafe_allow_html=True)
step_indicator()

# ============================================================
# STAGE 1: Upload — no column dropdowns, no topic config.
# ============================================================
if st.session_state.stage == "upload":
    st.subheader("Stage 1 — Data intake")
    st.caption("We'll protect personal data first, then find which self-service articles would cut the most repeat emails.")

    st.markdown('<div class="mission-card"><div><div class="mission-label">Engagement brief</div><div class="mission-title">Upload a mailbox export to establish the knowledge-demand baseline</div></div><div class="xp-pill">Stage 1 of 4</div></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="feature-grid">
      <div class="feature-card"><div class="feature-icon">01</div><div class="feature-title">Privacy by design</div><div class="feature-copy">Personal identifiers are protected before analytical processing.</div></div>
      <div class="feature-card"><div class="feature-icon">02</div><div class="feature-title">Traceable evidence</div><div class="feature-copy">Each recommendation is supported by real, redacted mailbox examples.</div></div>
      <div class="feature-card"><div class="feature-icon">03</div><div class="feature-title">Decision-ready output</div><div class="feature-copy">A clear, ranked content backlog focused on measurable demand.</div></div>
    </div>
    """, unsafe_allow_html=True)

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

        # A formatted worksheet can report dozens of rows even when those rows
        # contain no values. Remove structural blanks before validating it as a
        # mailbox or presenting an email count.
        df = df.replace(r"^\s*$", pd.NA, regex=True).dropna(how="all").reset_index(drop=True)
        if len(df) == 0:
            st.error("This file is readable, but it contains no populated rows to analyze.")
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
        detected, confident = detect_all_columns(df)
        valid_mailbox, validation_message = validate_mailbox_dataframe(df, detected, confident)
        if not valid_mailbox:
            st.error(validation_message)
            st.caption("Upload a mailbox export containing a subject field and a message/body field. Sender is optional.")
            st.stop()

        st.session_state.raw_df = df
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
        st.write("")
        if st.button("Start →", type="primary"):
            st.session_state.use_llm = False  # always starts with the free, automatic pass
            st.session_state.stage = "cleaning"
            st.rerun()

# ============================================================
# STAGE 2: Cleaning
# ============================================================
elif st.session_state.stage == "cleaning":
    st.subheader("Stage 2 — Data protection and preparation")
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
    st.success(f"**Preparation complete:** {st.session_state.pseudonym_count} unique names, emails, and phone numbers protected.")
    st.markdown('<div class="mission-card"><div><div class="mission-label">Control status</div><div class="mission-title">Personal data protected and reply history safely removed</div></div><div class="xp-pill">Validated</div></div>', unsafe_allow_html=True)

    # Show a record where redaction visibly changed the content. The first
    # mailbox row often contains no PII, which made the previous before/after
    # example look identical even when other rows were protected correctly.
    example_candidates = []
    for field, field_label in [(st.session_state.body_col, "Message body"),
                               (st.session_state.subject_col, "Subject line")]:
        redacted_field = field + "_redacted"
        if field not in working_df.columns or redacted_field not in working_df.columns:
            continue
        for idx in working_df.index:
            original = str(working_df.at[idx, field] or "")
            protected = str(working_df.at[idx, redacted_field] or "")
            if original != protected:
                token_types = re.findall(r"\[(PERSON|EMAIL_ADDRESS|PHONE_NUMBER)_\d+\]", protected)
                # Prefer an unmistakable, explanatory example: email and phone
                # redactions are clearer than a name alone; a message body is
                # generally more understandable than an isolated subject.
                score = (token_types.count("EMAIL_ADDRESS") * 100 +
                         token_types.count("PHONE_NUMBER") * 90 +
                         token_types.count("PERSON") * 20 +
                         (5 if field_label == "Message body" else 0) +
                         min(len(original), 300) / 100)
                example_candidates.append((score, field_label, original, protected))

    example = max(example_candidates, key=lambda item: item[0])[1:] if example_candidates else None

    with st.expander("Review a redaction example", expanded=example is not None):
        if example:
            field_label, original, protected = example
            protected_html = html.escape(protected[:350])
            protected_html = re.sub(
                r"(\[(?:PERSON|EMAIL_ADDRESS|PHONE_NUMBER)_\d+\])",
                r'<span class="redaction-token">\1</span>', protected_html)
            st.caption(f"Example selected from: {field_label}. The original is displayed locally for comparison only.")
            c1, c2 = st.columns(2)
            c1.markdown("**Cleaned original**")
            c1.markdown(f'<div class="evidence-box">{html.escape(original[:350])}</div>', unsafe_allow_html=True)
            c2.markdown("**Protected version used for analysis**")
            c2.markdown(f'<div class="evidence-box">{protected_html}</div>', unsafe_allow_html=True)
        else:
            st.info("No subject or message-body text changed during redaction, so there is no meaningful before-and-after example to display.")

    if st.button("Continue →", type="primary"):
        st.session_state.stage = "scoping" if st.session_state.get("use_llm") else "analyzing"
        st.rerun()

# ============================================================
# STAGE 2.5: Scoping — only for AI-powered mode. Shows real cost/time
# for the full dataset, and lets the user choose a smaller, smartly
# stratified sample instead, with live-updating cost/time as they adjust.
# ============================================================
elif st.session_state.stage == "scoping":
    st.subheader("Stage 3 — Analysis scope")
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
    st.caption("Estimate includes semantic classification and a draft pack for up to three leading knowledge opportunities. Actual API usage may vary.")

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
                         width="stretch", hide_index=True)

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
    st.subheader("Stage 3 — Pattern analysis")
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
            # Turn the highest-value semantic findings into practical output.
            # Drafting is deliberately isolated from classification: if the
            # final writing call fails, the completed analysis remains usable.
            st.session_state.ai_draft_articles = []
            st.session_state.pop("ai_draft_warning", None)
            article_briefs = [
                {
                    "topic": row["topic"],
                    "demand_count": int(row["genuine_questions"]),
                    "examples": list(row["examples"]),
                }
                for _, row in backlog.head(3).iterrows()
                if int(row["genuine_questions"]) > 0
            ]
            if article_briefs:
                status.text("Drafting copy-ready knowledge articles for the leading opportunities...")
                try:
                    drafting_client = anthropic.Anthropic(api_key=st.session_state.api_key)
                    st.session_state.ai_draft_articles = draft_kb_articles_with_llm(
                        drafting_client, st.session_state.llm_model, article_briefs)
                except Exception as draft_error:
                    st.session_state.ai_draft_warning = str(draft_error)
            st.session_state.sample_size_used = len(data_to_analyze)
            st.session_state.population_size = len(st.session_state.working_df)
            status.empty()
            draft_count = len(st.session_state.ai_draft_articles)
            completion_note = f" and {draft_count} knowledge article drafts were prepared" if draft_count else ""
            st.success(f"Analysis complete — topics were discovered directly from this mailbox{completion_note}.")
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
        st.success("Analysis complete, using a general-purpose starting taxonomy. You can refine the topic categories from the results screen.")

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
    # Results use a dedicated view module to keep this analysis file focused
    # and to present the report as a concise, tabbed decision workspace.
    from results_view import render_results_page
    render_results_page(LLM_AVAILABLE, LLM_MODELS, confidence_band, suggest_format)
    st.stop()

    has_ai = "ai_backlog" in st.session_state
    if has_ai:
        view_choice = st.radio(
            "Which results to view?",
            ["AI-powered (more accurate)", "Fast / keyword-based"],
            horizontal=True,
            index=0 if st.session_state.get("active_result") == "ai" else 1,
        )
        active = "ai" if view_choice.startswith("AI-powered") else "free"
    else:
        active = "free"

    analyzed_df = st.session_state[f"{active}_analyzed_df"]
    backlog = st.session_state[f"{active}_backlog"]
    type_summary = st.session_state[f"{active}_type_summary"]
    total = len(analyzed_df)
    genuine_total = int((analyzed_df["email_type"] == "genuine_question").sum())
    opportunity_score = round(100 * genuine_total / total)

    st.subheader("Stage 4 — Executive findings")

    analysis_method = "AI-powered semantic classification" if active == "ai" else "Local keyword and rule-based classification"
    is_sample = active == "ai" and st.session_state.get("chosen_scope") == "sample"
    population_size = st.session_state.get("population_size", total) if is_sample else total

    st.markdown("## Purpose")
    st.write(
        "To identify recurring knowledge demand in the mailbox and prioritise the help content "
        "most likely to address genuine, repeatable questions."
    )

    st.markdown("## Scope of Analysis")
    scope_c1, scope_c2, scope_c3 = st.columns(3)
    scope_c1.metric("Mailbox population", population_size)
    scope_c2.metric("Records assessed", total)
    scope_c3.metric("Analysis method", "Semantic AI" if active == "ai" else "Rules-based")
    if is_sample:
        st.info(f"**Sample-based assessment:** {total} of {population_size} emails were classified. Findings are directional estimates, not exact whole-mailbox counts.")
    else:
        st.caption(f"The assessment covered all {total} records selected during data intake using {analysis_method.lower()}.")

    included_col, excluded_col = st.columns(2)
    with included_col:
        st.markdown("## Included in the Analysis")
        st.markdown(
            "- Redacted subject lines and current-message content\n"
            "- Classification of questions, acknowledgments, updates and automated messages\n"
            "- Topic demand, supporting examples and content priorities\n"
            + ("- Generic versus case-specific question assessment" if active == "ai" else "- Editable topic taxonomy and keyword patterns")
        )
    with excluded_col:
        st.markdown("## Excluded from the Analysis")
        excluded_items = [
            "Personal identifiers removed during preparation",
            "Quoted reply history and HTML formatting",
            "Attachments and information outside the uploaded export",
        ]
        if is_sample:
            excluded_items.append(f"Individual classification of {population_size - total} non-sampled emails")
        if active == "free":
            excluded_items.append("Semantic meaning and case-specificity assessment")
        st.markdown("\n".join(f"- {item}" for item in excluded_items))

    st.markdown("## Key Findings")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Emails analyzed", total)
    c2.metric("Genuine questions found", genuine_total, f"{100*genuine_total/total:.0f}% of mailbox")
    c3.metric("Topics identified", len(backlog))
    c4.metric("Names/emails/phones protected", st.session_state.pseudonym_count)

    st.markdown("### Knowledge Demand Summary")
    st.markdown(f"""
    <div class="achievement-grid">
      <div class="achievement"><div class="achievement-icon">PII</div><div><strong>Privacy control</strong><span>Identifiers protected before analysis</span></div></div>
      <div class="achievement"><div class="achievement-icon">MAP</div><div><strong>Demand profile</strong><span>{len(backlog)} knowledge gaps identified</span></div></div>
      <div class="achievement"><div class="achievement-icon">KB</div><div><strong>Content priorities</strong><span>Backlog supported by source evidence</span></div></div>
    </div>
    <div class="score-panel">
      <div class="score-ring" style="--score:{opportunity_score}"><div class="score-value">{opportunity_score}</div></div>
      <div class="score-copy"><strong>Self-Service Opportunity Score</strong><span>The share of this mailbox that looks like a genuine, answerable question. Treat this as the opportunity ceiling—not a promise that every email will disappear.</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    st.markdown("#### Mailbox composition")
    st.bar_chart(type_summary.set_index("type")["count"])

    st.write("")
    st.markdown("#### Prioritised knowledge article backlog")
    st.caption("Ranked by genuine, answerable question volume per topic — not raw email count.")

    if len(backlog) == 0:
        st.warning(
            "**No topics matched this mailbox.** The starting topic list is tuned for an IT "
            "Helpdesk mailbox (password resets, VPN, etc.) — if this is a different kind of "
            "mailbox, open **'Refine topic categories and re-run'** below and replace the "
            "topic list with keywords that match what this mailbox is actually about, then "
            "re-run the analysis."
        )

    medals = {1: "01", 2: "02", 3: "03"}
    for _, row in backlog.iterrows():
        label, cls, dot = confidence_band(row["pct_genuine"])
        medal = medals.get(row["rank"], f'{row["rank"]}')
        fmt = suggest_format(row["topic"])

        st.markdown(f"""
        <div class="article-card {'rank-one' if row['rank'] == 1 else ''}">
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

    st.markdown("## Assumptions and Limitations")
    limitations = [
        "Results reflect only the content and data quality of the uploaded mailbox export.",
        "Rule-based results depend on the configured topic terms and may miss context, synonyms or unfamiliar domains.",
        "Redaction and quoted-reply removal are automated controls; representative records should be reviewed before decisions are finalised.",
        "The opportunity score is an indicator of potentially answerable demand, not a forecast of contact reduction.",
    ]
    if is_sample:
        limitations.append("Sample-based topic volumes are estimates and should not be presented as exact full-mailbox counts.")
    if active == "ai":
        limitations.append("AI classifications may contain errors and should be validated by a subject-matter owner.")
    st.markdown("\n".join(f"- {item}" for item in limitations))

    topic_report_lines = [
        f"{int(row['rank'])}. **{row['topic']}** — {int(row['genuine_questions'])} priority questions from {int(row['total_emails'])} topic emails."
        for _, row in backlog.iterrows()
    ] or ["No topic demand met the configured matching criteria."]
    report_markdown = f"""# Mailbox Knowledge Gap Assessment

## Purpose
To identify recurring knowledge demand in the mailbox and prioritise help content most likely to address genuine, repeatable questions.

## Scope of Analysis
- Mailbox population: {population_size}
- Records assessed: {total}
- Method: {analysis_method}
- Scope basis: {'Stratified sample; findings are directional estimates' if is_sample else 'All records selected during data intake'}

## Included in the Analysis
- Redacted subject lines and current-message content
- Email type and topic classification
- Knowledge-demand ranking and supporting evidence

## Excluded from the Analysis
{chr(10).join(f'- {item}' for item in excluded_items)}

## Key Findings
- Genuine questions identified: {genuine_total} ({100*genuine_total/total:.0f}% of assessed records)
- Topics identified: {len(backlog)}
- Self-Service Opportunity Score: {opportunity_score}/100

### Knowledge Demand Summary
{chr(10).join(topic_report_lines)}

## Assumptions and Limitations
{chr(10).join(f'- {item}' for item in limitations)}
"""

    dl1, dl2 = st.columns(2)
    dl1.download_button("Download structured report", report_markdown.encode("utf-8"), "mailbox_knowledge_gap_report.md", "text/markdown", width="stretch")
    dl2.download_button("Download backlog data", backlog.drop(columns=["examples"]).to_csv(index=False).encode("utf-8"), "knowledge_backlog.csv", "text/csv", width="stretch")

    if not has_ai:
        st.write("")
        st.divider()
        st.markdown("""
        <section class="upgrade-panel">
          <div class="upgrade-eyebrow">Semantic intelligence</div>
          <div class="upgrade-title">Move from keyword matches to decision-grade insight</div>
          <div class="upgrade-copy">The baseline identifies obvious patterns. Semantic analysis reads the meaning of each message, discovers the mailbox's own topic structure and isolates the questions a knowledge article can genuinely resolve.</div>
          <div class="upgrade-grid">
            <div class="upgrade-benefit"><strong>Discover topics automatically</strong><span>No fixed taxonomy. The model derives themes from the actual mailbox.</span></div>
            <div class="upgrade-benefit"><strong>Understand intent and context</strong><span>Distinguishes real questions from updates, acknowledgments and automated traffic.</span></div>
            <div class="upgrade-benefit"><strong>Prioritise addressable demand</strong><span>Separates reusable guidance from requests that require a personal case lookup.</span></div>
          </div>
        </section>
        <table class="comparison">
          <thead><tr><th>Capability</th><th>Baseline analysis</th><th>Semantic analysis</th></tr></thead>
          <tbody>
            <tr><td>Topic model</td><td>Pre-set keyword rules</td><td>Discovered from mailbox content</td></tr>
            <tr><td>Message understanding</td><td>Literal phrase matching</td><td>Meaning, intent and context</td></tr>
            <tr><td>Knowledge opportunity</td><td>All detected questions</td><td>Generic, article-addressable questions</td></tr>
            <tr><td>Best use</td><td>Immediate directional baseline</td><td>Defensible content investment decisions</td></tr>
          </tbody>
        </table>
        """, unsafe_allow_html=True)
        if not LLM_AVAILABLE:
            st.info("Semantic analysis is unavailable in this deployment because the Anthropic client is not installed.")
        else:
            st.markdown('<div class="commercial-note"><strong>Controlled spend:</strong> the next step makes one small discovery call on approximately 40 redacted emails. You will then see the estimated full-run cost and time, with a lower-cost sample option, before authorising classification.</div>', unsafe_allow_html=True)
            setup_c1, setup_c2 = st.columns([1.35, 1])
            with setup_c1:
                api_key_input = st.text_input("Anthropic API key", type="password",
                                               placeholder="sk-ant-...",
                                               help="Used only for this browser session and never written to the repository.")
            with setup_c2:
                model_choice = st.selectbox("Analysis model", list(LLM_MODELS.keys()))
            if st.button("Preview semantic topics and cost", type="primary", disabled=not api_key_input, width="stretch"):
                st.session_state.api_key = api_key_input
                st.session_state.llm_model = LLM_MODELS[model_choice]
                st.session_state.use_llm = True
                st.session_state.pop("scoping_topics", None)
                st.session_state.stage = "scoping"
                st.rerun()
            st.caption("No full-mailbox analysis starts from this button. Scope and estimated spend are confirmed on the next screen.")

    # Topic refinement remains available as a secondary baseline control.
    st.write("")
    if active == "free":
        with st.expander("Advanced: refine baseline topic rules"):
            st.caption("Adjust keyword rules, add a missing topic or remove one that does not apply, then re-run without uploading or redacting the data again.")
            topics_df = pd.DataFrame([{"topic": k, "keyword_pattern": v} for k, v in st.session_state.topic_patterns.items()])
            edited = st.data_editor(topics_df, num_rows="dynamic", width="stretch", key="topic_editor")
            if st.button("Re-run baseline analysis"):
                st.session_state.topic_patterns = {r["topic"]: r["keyword_pattern"] for _, r in edited.iterrows()
                                                    if r["topic"] and r["keyword_pattern"]}
                st.session_state.use_llm = False
                st.session_state.stage = "analyzing"
                st.rerun()
    else:
        st.caption("Semantic topics were discovered from this mailbox; no fixed keyword taxonomy was used.")

    st.write("")
    if st.button("Start over with a new file"):
        for key in list(st.session_state.keys()):
            st.session_state.pop(key, None)
        st.rerun()
