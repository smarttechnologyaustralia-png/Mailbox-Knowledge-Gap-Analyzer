"""
Mailbox Knowledge Gap Analyzer — Smart Technology Inbox Audit tool
-----------------------------------------------------------
Upload -> Clean -> Analyze (smart defaults) -> Results (refine & re-run).
Column mapping and topic configuration are both hidden by default and
only surfaced when genuinely needed.

RUN:
    pip install -r requirements.txt   # includes the en_core_web_sm spaCy model
    streamlit run app.py
"""
import html
import re

import pandas as pd
import streamlit as st

try:
    import anthropic  # noqa: F401 -- presence check for the optional semantic path
    from llm_classifier import (discover_topics_with_llm, draft_kb_articles_with_llm,
                                make_client, estimate_cost, BATCH_SIZE)
    from scoping_logic import quick_keyword_prescan, estimate_time_and_cost, format_time, build_stratified_sample
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

LLM_MODELS = {
    "Fast & cheap (recommended)": "claude-haiku-4-5-20251001",
    "More capable, higher cost": "claude-sonnet-4-6",
}


def _server_api_key():
    """Analysis runs on Smart Techno's key (Streamlit secrets or env),
    never a customer-supplied one."""
    import os
    key = ""
    try:
        key = st.secrets.get("ANTHROPIC_API_KEY", "")
    except Exception:
        pass
    return key or os.environ.get("ANTHROPIC_API_KEY", "")


def _confidence_label(coverage):
    if coverage >= 0.999:
        return "Complete — every email verified"
    if coverage >= 0.35:
        return "High confidence — smart-sampled, all topics covered"
    if coverage >= 0.12:
        return "Good confidence — representative sample"
    return "Directional — small sample"

TIER_49_MAX = 500


st.set_page_config(page_title="Smart Technology | Mailbox Knowledge Gap Analyzer", layout="wide", page_icon="◆")

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
.brand-lockup { position:relative; z-index:1; display:flex; align-items:center; gap:11px; margin-bottom:27px; }
.brand-mark { width:38px; height:38px; flex:none; filter:drop-shadow(0 5px 12px rgba(0,0,0,.14)); }
.brand-name { color:white; font-size:15px; line-height:1.1; font-weight:790; letter-spacing:-.015em; }
.brand-name span { display:block; color:#8ee3dc; font-size:8px; font-weight:800; letter-spacing:.16em; text-transform:uppercase; margin-top:5px; }
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
.sensitive-mask { display:inline-block; background:#25384d; color:white; border-radius:4px; padding:0 6px; font-size:10px; font-weight:750; letter-spacing:.03em; }
.privacy-note { background:#e8f7f5; color:#126b5e; padding:12px 16px; border-radius:10px; font-size:13px; }
.mission-card { display:flex; align-items:center; justify-content:space-between; gap:18px; background:#ffffff; border:1px solid #dfe6ed; border-left:4px solid #147d76; border-radius:10px; padding:17px 20px; margin:8px 0 22px; }
.mission-label { color:#147d76; font-size:10px; font-weight:850; letter-spacing:.14em; text-transform:uppercase; }
.mission-title { color:#173456; font-weight:760; font-size:15px; margin-top:3px; }
.stage-pill { flex:none; background:#edf3f7; color:#34506d; padding:8px 12px; border-radius:6px; font-size:10px; font-weight:800; letter-spacing:.06em; text-transform:uppercase; }
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
.guided-strip { display:grid; grid-template-columns:repeat(3,1fr); background:white; border:1px solid #dfe6ed; border-radius:12px; overflow:hidden; margin:8px 0 22px; }
.guided-step { padding:13px 15px; border-right:1px solid #e5ebf0; color:#69798a; font-size:10px; line-height:1.45; }
.guided-step:last-child { border-right:0; }
.guided-step b { display:block; color:#173456; font-size:11px; margin-bottom:2px; }
.guided-step.recommended { background:#eef8f6; }
.compact-cta { display:flex; align-items:center; justify-content:space-between; gap:16px; background:#102b4c; color:white; border-radius:12px; padding:16px 18px; margin:20px 0 10px; }
.compact-cta strong { display:block; color:white; font-size:13px; }
.compact-cta span { color:#bfd0dc; font-size:10px; }
@media (prefers-reduced-motion: reduce) { .feature-card { transition:none; } }
@media (max-width: 720px) {
  .block-container { padding-top:1.5rem; }
  .hero { padding:26px 22px; border-radius:18px; }
  .feature-grid { grid-template-columns:1fr; }
  .upgrade-grid { grid-template-columns:1fr; }
  .report-cta { grid-template-columns:1fr; }
  .report-cta-status { width:max-content; }
  .guided-strip { grid-template-columns:1fr; }
  .guided-step { border-right:0; border-bottom:1px solid #e5ebf0; }
  .compact-cta { align-items:flex-start; }
  .score-panel { align-items:flex-start; }
  .steps { margin-bottom:30px; }
}

/* Results workspace — restrained executive-report treatment */
.results-hero {
  background:linear-gradient(125deg,#102f52 0%,#164c67 68%,#147d76 100%);
  border-radius:22px; padding:30px 34px; color:#fff; margin:8px 0 22px;
  box-shadow:0 18px 44px rgba(12,34,61,.16); position:relative; overflow:hidden;
}
.results-hero:after { content:""; position:absolute; width:210px; height:210px; border-radius:50%; right:-80px; top:-105px; background:rgba(255,255,255,.08); }
.results-hero .eyebrow { color:#70eadb; font-size:11px; font-weight:800; letter-spacing:1.8px; text-transform:uppercase; }
.results-hero h2 { color:#fff!important; font-size:31px!important; line-height:1.18!important; margin:10px 0 8px!important; max-width:760px; }
.results-hero p { margin:0; color:#dcebf1; max-width:780px; font-size:15px; line-height:1.65; }
.results-kpis { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:0 0 24px; }
.result-kpi { background:#fff; border:1px solid #dce5ec; border-radius:14px; padding:17px 18px; box-shadow:0 5px 16px rgba(12,34,61,.045); }
.result-kpi span { display:block; color:#66798c; font-size:11px; font-weight:750; letter-spacing:.65px; text-transform:uppercase; margin-bottom:7px; }
.result-kpi strong { display:block; color:#102f52; font-size:25px; line-height:1.1; }
.result-kpi small { display:block; color:#738398; font-size:12px; margin-top:5px; }
.results-guide { display:flex; align-items:center; justify-content:space-between; gap:16px; border:1px solid #d8e3ea; background:#f8fbfc; border-radius:12px; padding:12px 16px; margin:0 0 18px; color:#52697e; font-size:13px; }
.results-guide strong { color:#102f52; }
.article-library-head { display:flex; justify-content:space-between; align-items:flex-end; gap:18px; padding:4px 0 16px; border-bottom:1px solid #dce5ec; margin-bottom:16px; }
.article-library-head h3 { margin:0!important; }
.article-library-head span { color:#66798c; font-size:13px; }
.draft-summary { background:#fff; border:1px solid #dce5ec; border-left:4px solid #147d76; border-radius:12px; padding:16px 18px; margin:10px 0 6px; }
.draft-summary .draft-meta { color:#147d76; font-size:10px; font-weight:800; letter-spacing:1.2px; text-transform:uppercase; }
.draft-summary strong { display:block; color:#102f52; font-size:18px; margin:5px 0 4px; }
.draft-summary p { color:#5f7184; font-size:13px; margin:0; line-height:1.55; }
@media(max-width:800px){ .results-kpis{grid-template-columns:repeat(2,1fr)} .results-hero{padding:24px 22px} .results-guide{display:block} }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Core logic lives in analysis_core.py so it can be unit-tested and
# reused without importing Streamlit. This file is flow and UI only.
# ============================================================
from analysis_core import (
    DEFAULT_TOPICS, INDUSTRY_PACKS, patterns_for_industry, parse_upload, detect_all_columns, validate_mailbox_dataframe,
    strip_html, strip_quotes, redact_dataframe, run_analysis, run_llm_analysis,
    confidence_band, suggest_format,
)


@st.cache_data(show_spinner="Reading file...")
def load_upload_cached(file_bytes, filename, read_limit):
    """Parse once per unique (file, limit) pair. Streamlit reruns the whole
    script on every widget interaction; without this cache, each click of
    the large-file radio buttons re-read and re-parsed the entire upload."""
    return parse_upload(file_bytes, filename, read_limit)


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
  <div class="brand-lockup">
    <svg class="brand-mark" viewBox="0 0 64 64" aria-label="Smart Technology logo"><rect width="64" height="64" rx="15" fill="#07182C"/><path d="M16 22.5c0-5.2 4.3-9.5 9.5-9.5H42v8H25.5a1.5 1.5 0 0 0 0 3h8a9.5 9.5 0 0 1 0 19H17v-8h16.5a1.5 1.5 0 0 0 0-3h-8c-5.2 0-9.5-4.3-9.5-9.5Z" fill="#FFFFFF"/><path d="M39 13h10v38h-8V23.5L35 29v-10l4-3.6Z" fill="#43C7B5"/><circle cx="49" cy="51" r="4" fill="#43C7B5"/></svg>
    <div class="brand-name">Smart Technology<span>Clear insight. Better service.</span></div>
  </div>
  <div class="hero-kicker">Knowledge intelligence · privacy first</div>
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

    st.markdown('<div class="mission-card"><div><div class="mission-label">Engagement brief</div><div class="mission-title">Upload a mailbox export to establish the knowledge-demand baseline</div></div><div class="stage-pill">Stage 1 of 4</div></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="feature-grid">
      <div class="feature-card"><div class="feature-icon">01</div><div class="feature-title">Privacy by design</div><div class="feature-copy">Personal identifiers are protected before analytical processing.</div></div>
      <div class="feature-card"><div class="feature-icon">02</div><div class="feature-title">Traceable evidence</div><div class="feature-copy">Each recommendation is supported by real, redacted mailbox examples.</div></div>
      <div class="feature-card"><div class="feature-icon">03</div><div class="feature-title">Decision-ready output</div><div class="feature-copy">A clear, ranked content backlog focused on measurable demand.</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    class _LocalUpload:
        """Duck-typed stand-in for a Streamlit UploadedFile, used by the
        one-click sample so prospects can see results without providing data."""
        def __init__(self, path):
            with open(path, "rb") as fh:
                self._bytes = fh.read()
            self.name = path.split("/")[-1]
            self.size = len(self._bytes)
        def getvalue(self):
            return self._bytes

    industry_label = st.selectbox(
        "What kind of inbox is this?",
        list(INDUSTRY_PACKS.keys()),
        help="Tunes the free analysis to your industry's common question types. You can refine topics later.")
    st.session_state.topic_patterns = patterns_for_industry(industry_label)
    st.session_state.industry_label = industry_label

    demo_col, _ = st.columns([1, 2])
    with demo_col:
        if st.button("See it work on sample data", help="Loads a fictional 500-email helpdesk mailbox. Nothing is uploaded."):
            st.session_state.demo_upload = True
    st.caption("No mailbox export handy? The sample run uses a fictional helpdesk dataset — nothing leaves your browser session either way.")

    uploaded_file = st.file_uploader(
        "Mailbox export (.xlsx, .xls, or .csv)", type=["xlsx", "xls", "csv"],
        help="Maximum upload size: 500 MB. Files above 50 MB default to a 1,000-row assessment. For very large mailboxes, CSV is the most memory-safe format.")

    if not uploaded_file and st.session_state.get("demo_upload"):
        try:
            uploaded_file = _LocalUpload("Sample_IT_Helpdesk_Mailbox_500.xlsx")
        except OSError:
            st.error("Sample dataset is missing from this deployment.")
            st.session_state.demo_upload = False

    if uploaded_file:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        pre_sampled = False
        source_sample_method = None
        read_limit = None
        if file_size_mb > 50:
            st.warning(
                f"This file is {file_size_mb:.1f} MB. Large spreadsheet files can expand substantially in memory during processing."
            )
            large_file_choice = st.radio(
                "Large-file processing mode",
                ["Assess the first 1,000 rows (recommended)",
                 "Attempt to load the complete file (may exceed hosting memory)"],
                help="The recommended option avoids loading the entire workbook and clearly labels results as sample-based.")
            if large_file_choice.startswith("Assess"):
                read_limit = 1000
                pre_sampled = True
                source_sample_method = "the first 1,000 rows of a large file"
        try:
            df = load_upload_cached(uploaded_file.getvalue(), uploaded_file.name, read_limit)
        except Exception as e:
            # Generic message to the user; raw library errors can echo file
            # internals and mean nothing to a non-technical audience.
            st.error("Couldn't read this file. Check it's a valid, unlocked .xlsx/.xls/.csv export.")
            with st.expander("Technical details"):
                st.code(str(e))
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
        source_population_size = None
        if len(df) > MAX_ROWS_DEFAULT:
            st.warning(f"This file has {len(df)} rows. For a fast first pass, analyzing a sample of "
                       f"the first {MAX_ROWS_DEFAULT} is much faster.")
            use_sample = st.radio("How much to analyze?",
                                   [f"Sample the first {MAX_ROWS_DEFAULT} rows (fast first pass)",
                                    f"Analyze all {len(df)} rows (accurate, may take a while)"])
            if use_sample.startswith("Sample"):
                source_population_size = len(df)
                df = df.sample(n=MAX_ROWS_DEFAULT, random_state=42).sort_index().copy()
                pre_sampled = True
                source_sample_method = "a reproducible random sample of 1,000 rows"
        detected, confident = detect_all_columns(df)
        valid_mailbox, validation_message = validate_mailbox_dataframe(df, detected, confident)
        if not valid_mailbox:
            st.error(validation_message)
            st.caption("Upload a mailbox export containing a subject field and a message/body field. Sender is optional.")
            st.stop()

        st.session_state.raw_df = df
        st.session_state.source_was_sampled = pre_sampled
        st.session_state.source_population_size = source_population_size
        st.session_state.source_sample_method = source_sample_method
        st.session_state.source_file_size_mb = round(file_size_mb, 1)
        st.session_state.subject_col = detected["subject"]
        st.session_state.body_col = detected["body"]
        st.session_state.sender_col = detected["sender"]

        sample_label = "sampled records" if pre_sampled else "emails"
        st.success(f"Loaded **{len(df)} {sample_label}** and identified the subject, message, and sender fields automatically.")
        if pre_sampled:
            st.caption("Results and exports will explicitly identify this as a file-level sample.")

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
        working_df, pseudonym_map, name_protection_active = redact_dataframe(
            prepared, [st.session_state.subject_col, body_col],
            progress_callback=lambda p: (progress.progress(0.15 + p * 0.85), status.text(f"Redacting personal details... {int(p*100)}%")))
        st.session_state.working_df = working_df
        st.session_state.pseudonym_count = len(pseudonym_map)
        st.session_state.name_protection_active = name_protection_active
    working_df = st.session_state.working_df
    status.empty()
    progress.progress(1.0)
    if st.session_state.get("name_protection_active", True):
        st.success(f"**Preparation complete:** {st.session_state.pseudonym_count} unique names, emails, and phone numbers protected.")
    else:
        st.warning(
            f"**Partial protection only:** name detection is unavailable in this deployment, so only "
            f"email addresses and phone numbers were protected ({st.session_state.pseudonym_count} unique values). "
            f"Personal names may remain in the text. Review the data before authorising any external analysis."
        )
    if st.session_state.get("name_protection_active", True):
        st.markdown('<div class="mission-card"><div><div class="mission-label">Control status</div><div class="mission-title">Personal data protected and reply history safely removed</div></div><div class="stage-pill">Validated</div></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="mission-card"><div><div class="mission-label">Privacy safeguard</div><div class="mission-title">Partial protection — semantic analysis disabled</div></div><div class="stage-pill">Standard only</div></div>', unsafe_allow_html=True)

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
            masked_original_html = re.sub(
                r"\[(?:PERSON|EMAIL_ADDRESS|PHONE_NUMBER)_\d+\]",
                r'<span class="sensitive-mask">SENSITIVE VALUE HIDDEN</span>',
                html.escape(protected[:350]))
            st.caption(f"Example selected from: {field_label}. Raw personal information is never displayed in the interface.")
            c1, c2 = st.columns(2)
            c1.markdown("**Before analysis (safe preview)**")
            c1.markdown(f'<div class="evidence-box">{masked_original_html}</div>', unsafe_allow_html=True)
            c2.markdown("**Protected version used for analysis**")
            c2.markdown(f'<div class="evidence-box">{protected_html}</div>', unsafe_allow_html=True)
        else:
            st.info("No subject or message-body text changed during redaction, so there is no meaningful before-and-after example to display.")

    st.caption("Recommended next step: continue to the free baseline analysis. No API key or external service is required.")
    back_col, continue_col = st.columns([1, 2.2])
    with back_col:
        if st.button("Back to upload", width="stretch"):
            # Clear EVERY analysis artifact, not just the raw data. Leaving
            # ai_*/free_* results behind meant a previous file's findings
            # could silently attach themselves to a newly uploaded file.
            for key in ["raw_df", "working_df", "pseudonym_count", "subject_col", "body_col", "sender_col",
                        "name_protection_active",
                        "free_analyzed_df", "free_backlog", "free_type_summary",
                        "ai_analyzed_df", "ai_backlog", "ai_type_summary", "active_result",
                        "scoping_topics", "chosen_scope", "chosen_sample_size",
                        "population_size", "sample_size_used",
                        "ai_draft_articles", "ai_draft_warning", "ai_failed_batches", "use_llm",
                        "source_was_sampled", "source_population_size",
                        "source_sample_method", "source_file_size_mb"]:
                st.session_state.pop(key, None)
            st.session_state.stage = "upload"
            st.rerun()
    with continue_col:
        if st.button("Continue to analysis", type="primary", width="stretch"):
            st.session_state.stage = "scoping" if st.session_state.get("use_llm") else "analyzing"
            st.rerun()

# ============================================================
# STAGE 2.5: Scoping — only for AI-powered mode. Shows real cost/time
# for the full dataset, and lets the user choose a smaller, smartly
# stratified sample instead, with live-updating cost/time as they adjust.
# ============================================================
elif st.session_state.stage == "scoping":
    if not st.session_state.get("name_protection_active", False):
        st.session_state.use_llm = False
        st.session_state.stage = "results" if "free_backlog" in st.session_state else "analyzing"
        st.warning("Semantic analysis was blocked because full name protection could not be verified. Continuing with privacy-safe standard analysis.")
        st.rerun()

    st.subheader("Stage 3 — Analysis scope")
    working_df = st.session_state.working_df
    subj_col = st.session_state.subject_col + "_redacted"
    body_col = st.session_state.body_col + "_redacted"
    total_n = len(working_df)

    if "scoping_topics" not in st.session_state:
        with st.spinner("Reading a small sample to see what topics are actually in this mailbox..."):
            server_key = _server_api_key()
            if not server_key:
                st.error("Verified analysis is temporarily unavailable — please contact hello@smarttechno.com.au and we'll run it for you.")
                st.stop()
            client = make_client(server_key)
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
    tier = st.session_state.get("tier", "49")
    tier_cap = total_n if tier == "99" else min(total_n, TIER_49_MAX)
    _, full_secs = estimate_time_and_cost(total_n, BATCH_SIZE, st.session_state.llm_model)

    if tier == "99":
        st.markdown(f"**Full Verification: {total_n} emails available** — estimated time **{format_time(full_secs)}**")
    else:
        st.markdown(f"**Verified Analysis: up to {tier_cap} of {total_n} emails** — smart-sampled so every topic gets fair coverage.")
        if total_n > TIER_49_MAX:
            st.caption(f"Want all {total_n} verified? Full Verification covers the whole export — contact hello@smarttechno.com.au to upgrade and we'll apply your $49.")
    st.caption("Verification includes semantic classification and a draft pack for up to three leading knowledge opportunities.")

    if tier == "99":
        choice = st.radio(
            "How much do you want verified?",
            [f"Verify all {total_n} emails (complete)",
             "Verify a smaller, representative sample (faster — every topic still gets fair coverage)"],
        )
    else:
        choice = "Verify a smaller, representative sample"

    if choice.startswith("Verify a smaller"):
        default_sample = min(tier_cap, max(100, int(total_n * 0.15)))
        slider_min = min(tier_cap, max(1, len(topics) * 5))
        default_sample = max(slider_min, default_sample)
        if tier_cap <= slider_min:
            sample_size = tier_cap
            st.caption(f"Verifying {sample_size} emails.")
        else:
            sample_size = st.slider("How many emails to verify", min_value=slider_min, max_value=tier_cap,
                                     value=default_sample, step=1 if tier_cap < 100 else 10)
        _, sample_secs = estimate_time_and_cost(sample_size, BATCH_SIZE, st.session_state.llm_model)
        coverage = sample_size / max(total_n, 1)
        st.markdown(f"**Selected: {sample_size} of {total_n} emails ({coverage:.0%} coverage)** — "
                    f"{_confidence_label(coverage)} — estimated time **{format_time(sample_secs)}**")

        preview_sample = build_stratified_sample(working_df, "_rough_topic", sample_size, min_per_topic=8)
        with st.expander("See how this sample would be split across topics"):
            st.dataframe(preview_sample["_rough_topic"].value_counts().rename_axis("topic").reset_index(name="sample_count"),
                         width="stretch", hide_index=True)

        st.session_state.chosen_scope = "sample"
        st.session_state.chosen_sample_size = sample_size
    else:
        st.session_state.chosen_scope = "full"

    st.write("")
    back_col, run_col = st.columns([1, 2.2])
    with back_col:
        if st.button("Back to results", width="stretch"):
            st.session_state.use_llm = False
            st.session_state.stage = "results"
            st.rerun()
    with run_col:
        if st.button("Confirm scope and run analysis", type="primary", width="stretch"):
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
            analyzed_df, backlog, type_summary, failed_batches = run_llm_analysis(
                data_to_analyze,
                st.session_state.subject_col + "_redacted", st.session_state.body_col + "_redacted",
                _server_api_key(), st.session_state.llm_model,
                progress_callback=progress.progress, status_callback=status.text,
                known_topics=st.session_state.get("scoping_topics"),
            )
            st.session_state.ai_failed_batches = failed_batches
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
                    drafting_client = make_client(_server_api_key())
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
            if failed_batches:
                st.warning(f"{failed_batches} classification batch(es) could not be completed and were "
                           f"recorded as 'unclear'. Affected counts are minimums; this is disclosed in the report.")
        except Exception as e:
            status.empty()
            st.error(f"AI-powered analysis failed: {e}")
            st.caption("Check your API key is valid and has available credit, or switch back to fast/rule-based mode by starting over.")
            st.stop()
    else:
        status.text("Classifying messages and ranking topic demand...")
        progress.progress(0.5)

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
