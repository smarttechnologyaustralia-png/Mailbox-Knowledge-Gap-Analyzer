"""Focused, tabbed results experience for the Streamlit application."""

import html

import pandas as pd
import streamlit as st


def _semantic_upgrade(llm_available, llm_models, key_prefix="semantic"):
    st.markdown("""
    <section class="upgrade-panel">
      <div class="upgrade-eyebrow">Semantic intelligence</div>
      <div class="upgrade-title">From mailbox evidence to a publishable knowledge programme</div>
      <div class="upgrade-copy">Semantic analysis converts recurring demand into a complete decision report and professionally structured article drafts that can move directly into editorial review.</div>
      <div class="upgrade-grid">
        <div class="upgrade-benefit"><strong>Full decision report</strong><span>Semantic findings, methodology, limitations and an evidence-backed priority roadmap.</span></div>
        <div class="upgrade-benefit"><strong>Copy-ready article drafts</strong><span>Up to three structured articles for the highest-value opportunities, ready for editorial review.</span></div>
        <div class="upgrade-benefit"><strong>Addressable demand</strong><span>Priorities are based on reusable questions, excluding requests that need personal lookup.</span></div>
      </div>
    </section>
    <table class="comparison">
      <thead><tr><th>Capability</th><th>Baseline</th><th>Semantic</th></tr></thead>
      <tbody>
        <tr><td>Topic model</td><td>Pre-set keyword rules</td><td>Discovered from mailbox content</td></tr>
        <tr><td>Understanding</td><td>Literal phrase matching</td><td>Meaning, intent and context</td></tr>
        <tr><td>Opportunity</td><td>All detected questions</td><td>Article-addressable questions</td></tr>
        <tr><td>Deliverable</td><td>Directional findings</td><td>Full report + drafted articles</td></tr>
        <tr><td>Best use</td><td>Directional baseline</td><td>Content investment decisions</td></tr>
      </tbody>
    </table>
    """, unsafe_allow_html=True)

    if not llm_available:
        st.info("Semantic analysis is unavailable because the Anthropic client is not installed in this deployment.")
        return

    st.markdown('<div class="commercial-note"><strong>Controlled spend:</strong> first, one small discovery call reviews approximately 40 redacted emails. Full-run cost and time are shown before classification is authorised.</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1.35, 1])
    with c1:
        api_key = st.text_input("Anthropic API key", type="password", placeholder="sk-ant-...", key=f"{key_prefix}_api_key",
                                help="Used only in this browser session and never written to the repository.")
    with c2:
        model_label = st.selectbox("Analysis model", list(llm_models.keys()), key=f"{key_prefix}_model")
    if st.button("Preview semantic topics and cost", type="primary", disabled=not api_key, width="stretch", key=f"{key_prefix}_submit"):
        st.session_state.api_key = api_key
        st.session_state.llm_model = llm_models[model_label]
        st.session_state.use_llm = True
        st.session_state.pop("scoping_topics", None)
        st.session_state.stage = "scoping"
        st.rerun()
    st.caption("No full-mailbox analysis starts here. Scope and estimated spend are confirmed on the next screen.")


def _draft_articles():
    drafts = st.session_state.get("ai_draft_articles", [])
    st.markdown("### Draft knowledge article pack")
    st.write("Evidence-grounded first drafts for the leading content opportunities. Review organisation-specific placeholders and policy details before publication.")

    warning = st.session_state.get("ai_draft_warning")
    if not drafts:
        if warning:
            st.warning("The semantic analysis completed, but article drafting could not be completed. The findings and backlog remain available.")
            with st.expander("Drafting error details"):
                st.code(warning)
        else:
            st.info("No article-addressable topics were available to draft.")
        return

    combined = ["# Draft Knowledge Article Pack", "", "> AI-assisted drafts. Validate policies, links and procedures before publication.", ""]
    for index, draft in enumerate(drafts, start=1):
        title = str(draft.get("title", "Untitled article"))
        audience = str(draft.get("audience", "General audience"))
        summary = str(draft.get("summary", ""))
        body = str(draft.get("body_markdown", ""))
        topic = str(draft.get("topic", ""))
        with st.container(border=True):
            st.caption(f"DRAFT {index:02d}  /  {topic.upper()}")
            st.markdown(f"## {title}")
            st.caption(f"Intended audience: {audience}")
            st.info(summary)
            st.markdown(body)
            with st.expander("Copy-ready Markdown"):
                st.code(f"# {title}\n\n{body}", language="markdown")
        combined.extend([f"# {title}", "", f"**Audience:** {audience}", "", summary, "", body, "", "---", ""])

    st.download_button("Download complete article pack", "\n".join(combined).encode("utf-8"),
                       "draft_knowledge_articles.md", "text/markdown", width="stretch")


def _priority_backlog(backlog, confidence_band, suggest_format):
    st.markdown("### Prioritised knowledge article backlog")
    st.caption("Ranked by genuine, answerable question demand—not raw email volume.")
    if backlog.empty:
        st.warning("No topics matched the current analysis. Review the baseline topic rules in Methodology & Export.")
        return

    for _, row in backlog.iterrows():
        label, cls, dot = confidence_band(row["pct_genuine"])
        rank = f"{int(row['rank']):02d}"
        topic = html.escape(str(row["topic"]))
        suggested_format = html.escape(suggest_format(str(row["topic"])))
        st.markdown(f"""
        <div class="article-card {'rank-one' if row['rank'] == 1 else ''}">
          <span style="font-size:20px;font-weight:800;color:#147d76;">{rank}</span>
          <span style="font-size:18px;font-weight:700;color:#173456;">&nbsp; {topic}</span>
          <span class="rank-badge {cls}" style="width:auto;border-radius:20px;padding:4px 12px;margin-left:10px;">{dot} {label} confidence</span>
          <br><br><b>{int(row['genuine_questions'])}</b> priority questions from <b>{int(row['total_emails'])}</b> emails on this topic
          ({row['pct_genuine']}% question concentration) &nbsp;&middot;&nbsp; recommended format: <b>{suggested_format}</b>
        </div>
        """, unsafe_allow_html=True)
        with st.expander(f"Supporting evidence — {row['topic']}"):
            if row["examples"]:
                for example in row["examples"]:
                    st.markdown(f'<div class="evidence-box">{html.escape(str(example))}</div>', unsafe_allow_html=True)
            else:
                st.caption("No clean example snippets are available for this topic.")


def render_results_page(llm_available, llm_models, confidence_band, suggest_format):
    has_ai = "ai_backlog" in st.session_state
    if has_ai:
        choice = st.radio("Result set", ["AI-powered", "Fast / keyword-based"], horizontal=True,
                          index=0 if st.session_state.get("active_result") == "ai" else 1)
        active = "ai" if choice == "AI-powered" else "free"
    elif has_ai:
        st.markdown("""
        <div class="guided-strip">
          <div class="guided-step"><b>1. Compare the baseline</b>Review the original directional findings.</div>
          <div class="guided-step"><b>2. Inspect rule matches</b>Check how keyword priorities were formed.</div>
          <div class="guided-step recommended"><b>3. View AI-powered results</b>Use the result-set control above to return to the full report.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        active = "free"

    analyzed_df = st.session_state[f"{active}_analyzed_df"]
    backlog = st.session_state[f"{active}_backlog"]
    type_summary = st.session_state[f"{active}_type_summary"]
    total = len(analyzed_df)
    genuine_total = int((analyzed_df["email_type"] == "genuine_question").sum())
    opportunity_score = round(100 * genuine_total / total) if total else 0
    method = "AI-powered semantic classification" if active == "ai" else "Local keyword and rule-based classification"
    is_sample = active == "ai" and st.session_state.get("chosen_scope") == "sample"
    population = st.session_state.get("population_size", total) if is_sample else total

    st.subheader("Stage 4 — Executive findings")
    if active == "ai":
        st.markdown("""
        <div class="guided-strip">
          <div class="guided-step"><b>1. Review findings</b>Confirm the opportunity and leading demand.</div>
          <div class="guided-step"><b>2. Validate priorities</b>Inspect the evidence behind each recommendation.</div>
          <div class="guided-step recommended"><b>3. Review article drafts</b>Complete placeholders before publishing.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="guided-strip">
          <div class="guided-step"><b>1. Review the baseline</b>Understand volume and obvious demand patterns.</div>
          <div class="guided-step"><b>2. Inspect priorities</b>Check the evidence in the backlog.</div>
          <div class="guided-step recommended"><b>3. Build the full report</b>Unlock semantic findings and drafted articles.</div>
        </div>
        """, unsafe_allow_html=True)

    if not has_ai:
        @st.dialog("Build the full semantic report", width="large")
        def full_report_dialog():
            _semantic_upgrade(llm_available, llm_models, key_prefix="semantic_dialog")

        def compact_report_cta(button_key):
            st.markdown('<div class="compact-cta"><div><strong>Ready for the decision-grade deliverable?</strong><span>Your protected dataset can now produce the full report and drafted article pack.</span></div><div class="report-cta-status">Dataset ready</div></div>', unsafe_allow_html=True)
            if st.button("Build the full report", type="primary", width="stretch", key=button_key):
                full_report_dialog()

    tab_names = ["Executive Summary", "Priority Backlog", "Methodology & Export"]
    if active == "ai":
        tab_names.append("Draft Articles")
    elif not has_ai:
        tab_names.append("Full Report & Article Drafting")
    tabs = st.tabs(tab_names)

    with tabs[0]:
        st.markdown("### Purpose")
        st.write("Identify recurring knowledge demand and prioritise the help content most likely to address genuine, repeatable questions.")
        if is_sample:
            st.info(f"Directional assessment based on a stratified sample of {total} from {population} mailbox records.")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Records assessed", total)
        c2.metric("Genuine questions", genuine_total, f"{100*genuine_total/total:.0f}% of assessed" if total else None)
        c3.metric("Knowledge topics", len(backlog))
        c4.metric("PII entities protected", st.session_state.get("pseudonym_count", 0))

        st.markdown(f"""
        <div class="score-panel">
          <div class="score-ring" style="--score:{opportunity_score}"><div class="score-value">{opportunity_score}</div></div>
          <div class="score-copy"><strong>Self-Service Opportunity Score</strong><span>Share of assessed records that appear to be genuine, answerable questions. This is an opportunity indicator, not a contact-reduction forecast.</span></div>
        </div>
        """, unsafe_allow_html=True)

        if not backlog.empty:
            leading = backlog.iloc[0]
            st.markdown('<div class="mission-label">Primary finding</div>', unsafe_allow_html=True)
            st.markdown(f"#### {html.escape(str(leading['topic']))} is the leading content opportunity")
            st.write(f"It contains {int(leading['genuine_questions'])} priority questions across {int(leading['total_emails'])} topic emails in the assessed dataset.")

        left, right = st.columns([1.1, 1])
        with left:
            st.markdown("#### Mailbox composition")
            st.bar_chart(type_summary.set_index("type")["count"])
        with right:
            st.markdown("#### Top demand areas")
            if backlog.empty:
                st.caption("No matched topics.")
            else:
                top = backlog.head(5)[["topic", "genuine_questions", "total_emails"]].rename(columns={
                    "topic": "Topic", "genuine_questions": "Priority questions", "total_emails": "Topic emails"})
                st.dataframe(top, hide_index=True, width="stretch")

        if not has_ai:
            st.markdown("""
            <section class="report-cta">
              <div>
                <div class="report-cta-label">Full semantic deliverable</div>
                <div class="report-cta-title">Turn this baseline into a publishable knowledge plan</div>
                <div class="report-cta-copy">Your mailbox is already cleaned and protected. Continue from the current analysis to produce the decision-ready deliverable without uploading or preparing the data again.</div>
                <div class="deliverable-line"><span>Full executive report</span><span>Semantic priority model</span><span>Up to 3 drafted articles</span><span>Copy-ready Markdown</span></div>
              </div>
              <div class="report-cta-status">Dataset ready</div>
            </section>
            """, unsafe_allow_html=True)

            if st.button("Build the full report", type="primary", width="stretch", key="open_full_report_dialog"):
                full_report_dialog()
            st.caption("One controlled API workflow. Topic preview, estimated cost and analysis scope are confirmed before the full run.")

    with tabs[1]:
        _priority_backlog(backlog, confidence_band, suggest_format)
        if not has_ai:
            compact_report_cta("open_full_report_from_backlog")

    with tabs[2]:
        st.markdown("### Scope of Analysis")
        s1, s2, s3 = st.columns(3)
        s1.metric("Mailbox population", population)
        s2.metric("Records assessed", total)
        s3.metric("Method", "Semantic AI" if active == "ai" else "Rules-based")
        st.caption("Sample-based directional findings." if is_sample else f"All records selected during data intake were assessed using {method.lower()}.")

        included, excluded = st.columns(2)
        with included:
            st.markdown("#### Included in the Analysis")
            st.markdown("- Redacted subjects and current-message content\n- Email type and topic classification\n- Demand ranking and supporting evidence" + ("\n- Generic versus case-specific assessment" if active == "ai" else ""))
        excluded_items = ["Personal identifiers removed during preparation", "Quoted reply history and HTML formatting", "Attachments and information outside the export"]
        if is_sample:
            excluded_items.append(f"Individual classification of {population-total} non-sampled emails")
        if active == "free":
            excluded_items.append("Semantic meaning and case-specificity assessment")
        with excluded:
            st.markdown("#### Excluded from the Analysis")
            st.markdown("\n".join(f"- {item}" for item in excluded_items))

        st.markdown("### Assumptions and Limitations")
        limitations = [
            "Findings reflect only the uploaded mailbox export and its data quality.",
            "Rules-based results depend on configured topic terms and may miss context or synonyms.",
            "Automated redaction and reply removal should be quality-checked on representative records.",
            "The opportunity score is not a forecast of contact reduction.",
        ]
        if is_sample:
            limitations.append("Sample-based volumes are estimates, not exact whole-mailbox counts.")
        if active == "ai":
            limitations.append("AI classifications require subject-matter validation.")
        st.markdown("\n".join(f"- {item}" for item in limitations))

        topic_lines = [f"{int(r['rank'])}. **{r['topic']}** — {int(r['genuine_questions'])} priority questions from {int(r['total_emails'])} topic emails." for _, r in backlog.iterrows()] or ["No topics met the matching criteria."]
        draft_appendix = ""
        if active == "ai" and st.session_state.get("ai_draft_articles"):
            article_sections = []
            for draft in st.session_state.ai_draft_articles:
                article_sections.append(
                    f"### {draft.get('title', 'Draft article')}\n\n"
                    f"**Audience:** {draft.get('audience', 'General audience')}\n\n"
                    f"{draft.get('summary', '')}\n\n{draft.get('body_markdown', '')}"
                )
            draft_appendix = "\n\n## Draft Knowledge Articles\n\n" + "\n\n---\n\n".join(article_sections)

        report = f"""# Mailbox Knowledge Gap Assessment

## Purpose
Identify recurring knowledge demand and prioritise help content for genuine, repeatable questions.

## Scope of Analysis
- Mailbox population: {population}
- Records assessed: {total}
- Method: {method}
- Basis: {'Stratified sample; directional estimates' if is_sample else 'All selected records'}

## Included in the Analysis
- Redacted subject and current-message content
- Email type and topic classification
- Demand ranking and supporting evidence

## Excluded from the Analysis
{chr(10).join(f'- {item}' for item in excluded_items)}

## Key Findings
- Genuine questions: {genuine_total} ({100*genuine_total/total:.0f}% of assessed records)
- Topics identified: {len(backlog)}
- Self-Service Opportunity Score: {opportunity_score}/100

### Knowledge Demand Summary
{chr(10).join(topic_lines)}

## Assumptions and Limitations
{chr(10).join(f'- {item}' for item in limitations)}
{draft_appendix}
"""
        d1, d2 = st.columns(2)
        report_label = "Download full semantic report" if active == "ai" else "Download baseline report"
        d1.download_button(report_label, report.encode("utf-8"), "mailbox_knowledge_gap_report.md", "text/markdown", width="stretch")
        d2.download_button("Download backlog data", backlog.drop(columns=["examples"]).to_csv(index=False).encode("utf-8"), "knowledge_backlog.csv", "text/csv", width="stretch")

        if not has_ai:
            compact_report_cta("open_full_report_from_methodology")

        if active == "free":
            with st.expander("Advanced: refine baseline topic rules"):
                topics_df = pd.DataFrame([{"topic": k, "keyword_pattern": v} for k, v in st.session_state.topic_patterns.items()])
                edited = st.data_editor(topics_df, num_rows="dynamic", width="stretch", key="topic_editor_v2")
                if st.button("Re-run baseline analysis", key="rerun_baseline_v2"):
                    st.session_state.topic_patterns = {r["topic"]: r["keyword_pattern"] for _, r in edited.iterrows() if r["topic"] and r["keyword_pattern"]}
                    st.session_state.use_llm = False
                    st.session_state.stage = "analyzing"
                    st.rerun()

    if active == "ai":
        with tabs[3]:
            _draft_articles()
    elif not has_ai:
        with tabs[3]:
            _semantic_upgrade(llm_available, llm_models, key_prefix="semantic_tab")

    st.write("")
    back_col, restart_col = st.columns([1.25, 1])
    with back_col:
        back_label = "Back to analysis scope" if active == "ai" else "Back to data preparation"
        if st.button(back_label, width="stretch", key="back_from_results"):
            st.session_state.use_llm = active == "ai"
            st.session_state.stage = "scoping" if active == "ai" else "cleaning"
            st.rerun()
    with restart_col:
        if st.button("Start over with a new file", width="stretch", key="start_over_v2"):
            for key in list(st.session_state.keys()):
                st.session_state.pop(key, None)
            st.rerun()
