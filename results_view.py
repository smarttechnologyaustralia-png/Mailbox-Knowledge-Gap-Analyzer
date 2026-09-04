"""Focused, tabbed results experience for the Streamlit application."""

import html

import pandas as pd
import streamlit as st


def _check_license_key(code, secrets):
    """Validate a per-sale Lemon Squeezy license key. Returns tier string
    ("49"/"99") on success, or an error message string on failure.
    Uses the public License API (no auth token needed client-side)."""
    try:
        import requests
        r = requests.post(
            "https://api.lemonsqueezy.com/v1/licenses/activate",
            headers={"Accept": "application/json"},
            data={"license_key": code,
                  "instance_name": st.session_state.get("lead_email", "inbox-analyzer")},
            timeout=10)
        j = r.json()
    except Exception:
        return "Couldn't reach the licensing service — try again in a minute."
    if not j.get("activated"):
        err = str(j.get("error") or "That key isn't valid.")
        if "activation limit" in err.lower():
            return "This key has already been used the maximum number of times."
        return err
    pid = str((j.get("meta") or {}).get("product_id", ""))
    if pid and pid == str(secrets.get("LS_PRODUCT_ID_99", "")):
        return_tier = "99"
    elif pid and pid == str(secrets.get("LS_PRODUCT_ID_49", "")):
        return_tier = "49"
    else:
        return "This key belongs to a different product."
    return return_tier


def _post_lead(payload):
    """Best-effort lead/intake POST to Formspree. Never blocks the user:
    endpoint unset or request failure just means the lead stays session-only."""
    endpoint = ""
    try:
        endpoint = st.secrets.get("FORMSPREE_ENDPOINT", "")
        if not endpoint:
            lead_email = st.secrets.get("LEAD_EMAIL", "")
            if lead_email:
                # FormSubmit: free, no-signup form-to-email relay. First
                # submission triggers a one-time activation email to LEAD_EMAIL.
                endpoint = f"https://formsubmit.co/ajax/{lead_email}"
    except Exception:
        pass
    if not endpoint:
        return False
    try:
        import requests
        requests.post(endpoint, data=payload, timeout=8)
        return True
    except Exception:
        return False

from analysis_core import sanitize_for_csv


def _semantic_upgrade(llm_available, llm_models, key_prefix="semantic"):
    """Paid verification paywall. Replaces the old bring-your-own-API-key form:
    the customer buys an unlock code via Stripe (shown on the payment
    confirmation page), enters it here, and the analysis runs on our key."""
    if not llm_available:
        st.warning("Verified analysis is temporarily unavailable in this deployment.")
        return

    TIER_49_LINK = st.secrets.get("PAY_LINK_49", st.secrets.get("STRIPE_LINK_49", "#"))
    TIER_99_LINK = st.secrets.get("PAY_LINK_99", st.secrets.get("STRIPE_LINK_99", "#"))
    AUDIT_LINK = "mailto:hello@smarttechno.com.au?subject=Inbox%20Audit%20enquiry"

    st.markdown("#### Choose how deep to go")
    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.markdown("**Verified Analysis — $49**")
            st.caption("AI verification of up to 500 emails (smart-sampled so every topic gets fair coverage), full ranked evidence, and up to 3 machine-drafted knowledge articles.")
            st.link_button("Buy — $49", TIER_49_LINK, width="stretch")
    with c2:
        with st.container(border=True):
            st.markdown("**Full Verification — $99**")
            st.caption("Every email in your export verified, complete ranked evidence, and the full draft article pack.")
            st.link_button("Buy — $99", TIER_99_LINK, width="stretch")
    with c3:
        with st.container(border=True):
            st.markdown("**Inbox Audit — $490**")
            st.caption("Everything above plus consultant review, the fix recommendation, and our guarantee: 5+ recoverable hours/week found or the fee comes back.")
            st.link_button("Talk to us", AUDIT_LINK, width="stretch")

    st.caption("After payment, your personal unlock key is emailed to you with your receipt. Each key works once.")
    u1, u2 = st.columns([2, 1])
    code = u1.text_input("Unlock code", key=f"{key_prefix}_unlock", placeholder="e.g. ST-XXXX",
                         label_visibility="collapsed")
    if u2.button("Unlock & continue", type="primary", width="stretch", key=f"{key_prefix}_submit"):
        code_norm = (code or "").strip()
        tier = None
        err_msg = None
        if code_norm and code_norm.upper() == str(st.secrets.get("UNLOCK_CODE_99", "")).upper():
            tier = "99"
        elif code_norm and code_norm.upper() == str(st.secrets.get("UNLOCK_CODE_49", "")).upper():
            tier = "49"
        elif code_norm and (st.secrets.get("LS_PRODUCT_ID_49", "") or st.secrets.get("LS_PRODUCT_ID_99", "")):
            result = _check_license_key(code_norm, st.secrets)
            if result in ("49", "99"):
                tier = result
            else:
                err_msg = result
        if tier is None:
            st.error(err_msg or "That code isn't valid. Check your receipt email, or contact hello@smarttechno.com.au.")
        else:
            st.session_state.tier = tier
            st.session_state.llm_model = "claude-haiku-4-5-20251001"
            st.session_state.use_llm = True
            st.session_state.pop("scoping_topics", None)
            st.session_state.stage = "scoping"
            st.rerun()
    st.caption("No full-mailbox analysis starts here. Scope is confirmed on the next screen.")


def _draft_articles():
    drafts = st.session_state.get("ai_draft_articles", [])
    warning = st.session_state.get("ai_draft_warning")
    if not drafts:
        st.markdown("### Draft knowledge article pack")
        if warning:
            st.warning("The semantic analysis completed, but article drafting could not be completed. The findings and backlog remain available.")
            with st.expander("Drafting error details"):
                st.code(warning)
        else:
            st.info("No article-addressable topics were available to draft.")
        return

    combined = ["# Smart Technology", "", "## Draft Knowledge Article Pack", "", "> Clear insight. Better service.", "", "> AI-assisted drafts. Validate policies, links and procedures before publication.", ""]
    for draft in drafts:
        combined.extend([
            f"# {draft.get('title', 'Untitled article')}", "",
            f"**Audience:** {draft.get('audience', 'General audience')}", "",
            str(draft.get("summary", "")), "", str(draft.get("body_markdown", "")), "", "---", "",
        ])

    st.markdown(f"""
    <div class="article-library-head">
      <div><h3>Draft knowledge article library</h3><span>{len(drafts)} evidence-grounded drafts ready for editorial review</span></div>
    </div>
    """, unsafe_allow_html=True)
    st.download_button("Download complete article pack", "\n".join(combined).encode("utf-8"),
                       "draft_knowledge_articles.md", "text/markdown", type="primary", width="stretch")
    st.caption("Review organisation-specific placeholders, policies and links before publishing. Open only the drafts you want to inspect.")

    for index, draft in enumerate(drafts, start=1):
        title = str(draft.get("title", "Untitled article"))
        audience = str(draft.get("audience", "General audience"))
        summary = str(draft.get("summary", ""))
        body = str(draft.get("body_markdown", ""))
        topic = str(draft.get("topic", ""))
        st.markdown(f"""
        <div class="draft-summary">
          <div class="draft-meta">Draft {index:02d} &nbsp;·&nbsp; {html.escape(topic)} &nbsp;·&nbsp; {html.escape(audience)}</div>
          <strong>{html.escape(title)}</strong>
          <p>{html.escape(summary)}</p>
        </div>
        """, unsafe_allow_html=True)
        with st.expander(f"Open draft {index:02d} — {title}"):
            st.markdown(body)
            with st.expander("Copy-ready Markdown"):
                st.code(f"# {title}\n\n{body}", language="markdown")


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
          ({row['pct_genuine']}% of this topic's volume) &nbsp;&middot;&nbsp; recommended format: <b>{suggested_format}</b>
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
    source_sampled = bool(st.session_state.get("source_was_sampled", False))
    source_population = st.session_state.get("source_population_size")
    source_sample_method = st.session_state.get("source_sample_method") or "a 1,000-row file-level sample"

    leading_topic = html.escape(str(backlog.iloc[0]["topic"])) if not backlog.empty else "No dominant topic identified"
    result_label = "Semantic assessment complete" if active == "ai" else "Baseline assessment complete"
    result_copy = (
        f"{genuine_total} reusable questions were identified across {len(backlog)} knowledge areas. "
        "Use the ranked evidence to decide what your knowledge team should publish first."
        if active == "ai" else
        f"The local assessment found {genuine_total} potential questions across {len(backlog)} matched areas. "
        "Use this directional baseline to review obvious demand patterns."
    )
    st.markdown(f"""
    <section class="results-hero">
      <div class="eyebrow">{result_label}</div>
      <h2>{leading_topic} is the leading knowledge opportunity</h2>
      <p>{result_copy}</p>
    </section>
    <div class="results-guide"><span><strong>Recommended path</strong> &nbsp; Review the summary → validate ranked evidence → open the draft article library</span><span>{'Decision-ready semantic view' if active == 'ai' else 'Directional baseline view'}</span></div>
    """, unsafe_allow_html=True)

    # ---------- Lead capture (light gate before full results) ----------
    # NOTE: session-only until a webhook is wired (Formspree / Apps Script).
    # Persisting to local CSV is unreliable on Streamlit Community Cloud.
    if not st.session_state.get("lead_email"):
        with st.container(border=True):
            st.markdown("**Where should we send your summary?** The full results open straight after — no verification step.")
            lc1, lc2 = st.columns([2, 1])
            candidate = lc1.text_input("Work email", key="lead_email_input",
                                       label_visibility="collapsed", placeholder="you@company.com.au")
            phone = st.text_input("Phone (optional)", key="lead_phone_input", label_visibility="collapsed",
                                  placeholder="Phone (optional) — add it if you'd like a free 15-min walkthrough of your results")
            if lc2.button("Open my results", type="primary", width="stretch"):
                import re as _re
                if _re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", (candidate or "").strip()):
                    st.session_state.lead_email = candidate.strip()
                    st.session_state.lead_phone = (phone or "").strip()
                    _post_lead({"email": st.session_state.lead_email,
                                "phone": st.session_state.lead_phone, "source": "inbox-scan"})
                    st.rerun()
                else:
                    st.warning("That doesn't look like a valid email address.")
            st.caption("Used once, to send this summary. No mailing list.")
        st.stop()

    # ---------- Audit economics: the numbers that matter ----------
    with st.expander("Adjust cost assumptions", expanded=False):
        e1, e2, e3 = st.columns(3)
        export_weeks = e1.number_input("Weeks this export covers", min_value=1, max_value=104,
                                       value=int(st.session_state.get("eco_weeks", 4)), key="eco_weeks")
        mins_per_reply = e2.number_input("Avg minutes per reply", min_value=1, max_value=60,
                                         value=int(st.session_state.get("eco_mins", 6)), key="eco_mins")
        hourly_cost = e3.number_input("Loaded hourly cost (AUD)", min_value=20, max_value=250,
                                      value=int(st.session_state.get("eco_rate", 45)), key="eco_rate")
        st.caption("Defaults are deliberately conservative. Every derived figure below recalculates from these three inputs and is labelled as an estimate.")

    # ---------- Internal vs external: decides which fix applies ----------
    own_domain = st.text_input(
        "Your business email domain (optional)", key="own_domain",
        placeholder="e.g. yourcompany.com.au",
        help="Used only to split staff questions from customer enquiries. Sender addresses stay redacted.")
    if own_domain:
        from analysis_core import sender_domain_split
        _int, _ext, _unk = sender_domain_split(
            analyzed_df,
            st.session_state.get("sender_col"), own_domain)
        _tot = max(_int + _ext, 1)
        st.markdown(f"""
        <div class="results-kpis" style="grid-template-columns:repeat(2,1fr);">
          <div class="result-kpi"><span>Internal — your own staff asking</span><strong>{_int / _tot:.0%}</strong><small>Points to a staff knowledge assistant</small></div>
          <div class="result-kpi"><span>External — customers and suppliers asking</span><strong>{_ext / _tot:.0%}</strong><small>Points to a drafted-reply answer system</small></div>
        </div>
        """, unsafe_allow_html=True)

    handled_hours_total = (genuine_total * mins_per_reply) / 60.0
    hours_per_week = handled_hours_total / max(export_weeks, 1)
    weekly_cost = hours_per_week * hourly_cost
    annual_cost = weekly_cost * 48

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
        if is_sample:
            st.info(f"Directional assessment based on a stratified sample of {total} from {population} mailbox records.")
        if source_sampled:
            source_context = f" from {source_population} detected rows" if source_population else " from a large source file whose complete row count was not loaded"
            st.warning(f"File-level sample: this assessment uses {source_sample_method}{source_context}. Findings are directional, not exact whole-file counts.")
        if active == "ai" and st.session_state.get("ai_failed_batches"):
            failed = st.session_state.ai_failed_batches
            st.warning(f"{failed} classification batch(es) could not be completed during the semantic run and were "
                       f"recorded as 'unclear'. Question counts for affected topics are minimums, not totals.")

        st.markdown(f"""
        <div class="results-kpis">
          <div class="result-kpi"><span>Records assessed</span><strong>{total:,}</strong><small>{'Sample-based view' if is_sample or source_sampled else 'Selected mailbox population'}</small></div>
          <div class="result-kpi"><span>Reusable questions</span><strong>{genuine_total:,}</strong><small>{100*genuine_total/total:.0f}% of assessed records</small></div>
          <div class="result-kpi"><span>Knowledge areas</span><strong>{len(backlog)}</strong><small>Ranked content opportunities</small></div>
          <div class="result-kpi"><span>Opportunity indicator</span><strong>{opportunity_score}/100</strong><small>Not a reduction forecast</small></div>
        </div>
        <div class="results-kpis" style="grid-template-columns:repeat(3,1fr);">
          <div class="result-kpi"><span>Estimated handling time</span><strong>{hours_per_week:.1f} hrs/week</strong><small>Answering these questions manually</small></div>
          <div class="result-kpi"><span>Estimated weekly cost</span><strong>${weekly_cost:,.0f}</strong><small>At ${hourly_cost}/hr loaded cost</small></div>
          <div class="result-kpi"><span>Estimated annual cost</span><strong>${annual_cost:,.0f}</strong><small>48 working weeks — adjust assumptions above</small></div>
        </div>
        """, unsafe_allow_html=True)

        if not backlog.empty:
            leading = backlog.iloc[0]
            st.info(f"**Recommended first action:** validate the {int(leading['genuine_questions'])} reusable questions supporting **{leading['topic']}**, then move its draft article into editorial review.")

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
                <div class="report-cta-copy">The free scan showed you the cost. The verified analysis shows you the fix: every question ranked with evidence, AI-verified topic coverage, and drafted knowledge articles your team can publish. From $49 — your mailbox is already cleaned and protected, nothing to re-upload.</div>
                <div class="deliverable-line"><span>Every question ranked, with evidence</span><span>AI-verified coverage</span><span>Up to 3 drafted articles</span><span>From $49 — or the $490 guaranteed audit</span></div>
              </div>
              <div class="report-cta-status">Dataset ready</div>
            </section>
            """, unsafe_allow_html=True)

            if st.button("Build the full report", type="primary", width="stretch", key="open_full_report_dialog"):
                full_report_dialog()

            try:
                with open("SAMPLE_REPORT.md", "r", encoding="utf-8") as fh:
                    st.download_button("See a sample report first (free)", fh.read().encode("utf-8"),
                                       file_name="SmartTechno_Sample_Inbox_Audit.md", mime="text/markdown",
                                       width="stretch", key="sample_report_dl")
                st.caption("Fictional demonstration data — shows exactly what the paid report contains.")
            except OSError:
                pass
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
        if source_sampled:
            st.warning(f"Source coverage is limited to {source_sample_method}. The unprocessed portion of the source file is outside this assessment.")

        included, excluded = st.columns(2)
        with included:
            st.markdown("#### Included in the Analysis")
            st.markdown("- Redacted subjects and current-message content\n- Email type and topic classification\n- Demand ranking and supporting evidence" + ("\n- Generic versus case-specific assessment" if active == "ai" else ""))
        excluded_items = ["Personal identifiers removed during preparation", "Quoted reply history and HTML formatting", "Attachments and information outside the export"]
        if is_sample:
            excluded_items.append(f"Individual classification of {population-total} non-sampled emails")
        if source_sampled:
            excluded_items.append("Rows outside the file-level ingestion sample")
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
        if source_sampled:
            limitations.append("The source file was sampled during ingestion; findings cannot be treated as exact whole-file totals.")
        if active == "ai":
            limitations.append("AI classifications require subject-matter validation.")
            failed = st.session_state.get("ai_failed_batches", 0)
            if failed:
                limitations.append(f"{failed} classification batch(es) failed during the semantic run and were recorded as 'unclear'; affected counts are minimums.")
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

        eco_line = ""
        if st.session_state.get("eco_weeks"):
            _w = int(st.session_state.eco_weeks); _m = int(st.session_state.eco_mins); _r = int(st.session_state.eco_rate)
            _hrs = (genuine_total * _m) / 60.0 / max(_w, 1)
            eco_line = (f"\n**Estimated manual handling: {_hrs:.1f} hours/week "
                        f"(~${_hrs * _r:,.0f}/week, ~${_hrs * _r * 48:,.0f}/year) "
                        f"at {_m} min/reply and ${_r}/hr over a {_w}-week export. "
                        f"Figures are estimates from stated assumptions, not measurements.**\n")
        report = f"""# Smart Technology

{eco_line}
*Clear insight. Better service.*

## Mailbox Knowledge Gap Assessment

## Purpose
Identify recurring knowledge demand and prioritise help content for genuine, repeatable questions.

## Scope of Analysis
- Mailbox population: {population}
- Records assessed: {total}
- Method: {method}
- Basis: {'Stratified AI sample; directional estimates' if is_sample else 'All ingested records'}{' plus file-level sampling' if source_sampled else ''}

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
        d2.download_button("Download backlog data", sanitize_for_csv(backlog.drop(columns=["examples"])).to_csv(index=False).encode("utf-8"), "knowledge_backlog.csv", "text/csv", width="stretch")

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

    # ---------- Pain-point intake: the Offer B funnel ----------
    st.write("")
    with st.container(border=True):
        st.markdown("**What's the most manual, annoying process in your business right now?**")
        st.caption("Inbox or not — describe it in a sentence or two. We'll tell you straight whether it's fixable, and whether AI is even the right tool.")
        pain = st.text_area("Describe the problem", key="pain_point_input", label_visibility="collapsed",
                            placeholder="e.g. Every month-end we spend two days copying numbers between systems...")
        if st.button("Send it to us", key="pain_point_submit"):
            if (pain or "").strip():
                sent = _post_lead({"email": st.session_state.get("lead_email", ""),
                                   "phone": st.session_state.get("lead_phone", ""),
                                   "pain_point": pain.strip(), "source": "pain-point-intake"})
                st.success("Got it — we'll come back to you within one business day."
                           if sent else
                           "Got it — noted for this session. (Direct line: hello@smarttechno.com.au)")
            else:
                st.warning("Tell us the problem first — one sentence is enough.")

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
