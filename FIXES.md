# Remediation changelog — August 2026 review

Every finding from the code review, and what changed. Severity codes match
the review (C = critical, H = high, M = medium, L = low).

## Critical

- **C1 — Silent PII downgrade.** `redact_dataframe` now returns whether full
  name detection ran. When the fallback (email + phone only) is active, the
  preparation screen shows a partial-protection warning instead of a false
  "all names protected" success message, and a second warning appears before
  the semantic (external API) step. Regression-tested in
  `tests/test_core.py::TestRedactionFallback`.
- **C2 — Cross-file result contamination.** Returning to upload now clears
  every analysis artifact (`free_*`, `ai_*`, scope, sampling, drafts,
  failure counts), so a previous file's semantic results can no longer
  attach themselves to a newly uploaded file.
- **C3 — Invalid model ID.** `claude-sonnet-5` (nonexistent) replaced with
  `claude-sonnet-4-6`; the higher-tier semantic option now works.

## High

- **H1 — Excel memory safety.** `.xlsx` files are now streamed through
  openpyxl in read-only mode with a hard row cap (`parse_upload` in
  `analysis_core.py`) instead of pandas materialising the whole workbook.
  Uploader help text now honestly recommends CSV for very large mailboxes.
- **H2 — Reproducibility and output validation.** Classification and topic
  discovery run at `temperature=0`; every model classification is normalised
  against a strict vocabulary (`normalise_classification`) so variants like
  "Genuine Question" can no longer silently deflate counts.
- **H3 — Silent batch-failure degradation.** `run_llm_analysis` counts
  failed batches and returns the count; a non-zero count is disclosed as a
  warning on completion, on the Executive Summary, and in the downloadable
  report's Assumptions and Limitations.
- **H4 — Metric mismatch.** In the semantic path, the displayed percentage
  and confidence badge now use the same numerator as the ranked count
  (article-addressable questions); the backlog card copy was updated to
  match.
- **H5 — Dependency floor.** `streamlit>=1.48,<2` — the previous 1.40 floor
  permitted versions without the `width=` / `st.dialog` APIs this code uses.
- **H6 — Repeated file parsing.** Upload parsing is wrapped in
  `st.cache_data`, so widget interactions on the upload screen no longer
  re-read the entire file.

## Medium

- **M1 — CSV formula injection.** Exported backlog CSVs pass through
  `sanitize_for_csv`, escaping values beginning with `= + - @` (topic names
  originate from the LLM / untrusted mailbox content). Also fixed to work
  under both pandas 2.x and 3.x string dtypes.
- **M2 — Prompt injection.** All mailbox content entering prompts is
  delimited as data with an explicit ignore-embedded-instructions note
  (`DATA_BOUNDARY_NOTE`).
- **M3 — Overlapping redaction spans.** Overlapping Presidio results are
  resolved (highest score wins) before replacement, preventing corrupted
  hybrid placeholders.
- **M4 — Column detection collision.** Confidently named columns are
  excluded before content-based fallbacks run; subject and body can no
  longer resolve to the same column. Regression-tested.
- **M5 — Misclassified human replies.** "service desk" removed from
  automated-sender markers; replies written by real support staff are no
  longer counted as automated traffic.
- **M6 — API robustness.** All API clients are built via
  `make_client` with a 60s timeout and 2 retries; a hung call can no longer
  freeze the interface indefinitely.
- **M7 — Weak validation on named columns.** A minimal message-likeness
  check now applies even when subject/body columns match by name, so a
  tracker with a coincidental "Subject" column is rejected.
- **M8 — Dead code.** ~250 unreachable lines removed from `app.py` after the
  results handoff; unreachable branch removed from `results_view.py`;
  analysis logic extracted to `analysis_core.py` (UI-free, testable) —
  `app.py` is now flow and presentation only.

## Low

- Fake staged progress (`time.sleep` theater) removed from the free path.
- Docstring/README corrections (spaCy model name, sample-file reference).
- File-read errors show a generic message with technical details in an
  expander instead of echoing raw library errors.
- Gamification-era CSS (`xp-pill`, `achievement-*`) renamed or removed.
- API pricing consolidated into one dated constants block
  (`PRICING_PER_MTOK`).

## Tests

`tests/test_core.py` — 36 tests covering quote stripping, classification,
JSON extraction, output normalisation, stratified sampling, mailbox
validation, column detection, CSV sanitisation, redaction fallback, file
parsing, and end-to-end count reconciliation. Runs in under a second with
no Streamlit, no API access, and no model downloads.
