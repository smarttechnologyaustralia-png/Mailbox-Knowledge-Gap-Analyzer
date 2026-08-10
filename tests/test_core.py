"""
Unit tests for the Mailbox Knowledge Gap Analyzer core logic.

Run from the project root with either:
    python -m unittest discover -s tests -v
    python -m pytest tests/ -v          (if pytest is installed)

These tests deliberately avoid Streamlit, the Anthropic API, and the
Presidio/spaCy model download, so they run anywhere in under a second.
Each test that guards a fixed defect names the finding it protects.
"""
import io
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import analysis_core as core
import llm_classifier as llm
import scoping_logic


class TestStripQuotes(unittest.TestCase):
    def test_keeps_new_message_before_quote(self):
        body = "Hi, how do I reset my password? From: Someone Sent: yesterday old text"
        self.assertEqual(core.strip_quotes(body), "Hi, how do I reset my password?")

    def test_quote_before_content_keeps_content(self):
        # Regression guard: quote marker appearing FIRST must not delete
        # the real message that follows it.
        body = "From: helpdesk\nPlease find my earlier note attached, can you confirm receipt of the form?"
        result = core.strip_quotes(body)
        self.assertIn("confirm receipt", result)

    def test_no_marker_returns_unchanged(self):
        body = "Just a plain message with no reply history."
        self.assertEqual(core.strip_quotes(body), body)


class TestClassifyEmailType(unittest.TestCase):
    def test_question(self):
        self.assertEqual(core.classify_email_type("How do I connect to the VPN?", "Sam"), "genuine_question")

    def test_acknowledgment(self):
        self.assertEqual(core.classify_email_type("Thanks, all sorted now.", "Sam"), "acknowledgment")

    def test_automated_sender(self):
        self.assertEqual(core.classify_email_type("Your ticket was updated", "no-reply@corp"), "automated")

    def test_human_service_desk_reply_not_automated(self):
        # M5 regression: replies written by real service-desk staff must not
        # be classified as automated traffic.
        result = core.classify_email_type("How can we help you further?", "IT Service Desk")
        self.assertNotEqual(result, "automated")


class TestExtractJson(unittest.TestCase):
    def test_plain_array(self):
        self.assertEqual(llm._extract_json('["A", "B"]'), ["A", "B"])

    def test_fenced_array(self):
        self.assertEqual(llm._extract_json('```json\n["A", "B"]\n```'), ["A", "B"])

    def test_prefixed_text(self):
        self.assertEqual(llm._extract_json('Here are the topics:\n["A", "B"]'), ["A", "B"])

    def test_truncated_json_raises(self):
        with self.assertRaises(Exception):
            llm._extract_json('[{"type": "genuine_ques')


class TestNormaliseClassification(unittest.TestCase):
    # H2 regression: model output variants must land on the exact
    # vocabulary the pipeline compares against.
    def test_capitalised_type_normalised(self):
        result = llm.normalise_classification(
            {"type": "Genuine Question", "topic": "Billing", "specificity": "Generic"},
            ["Billing"])
        self.assertEqual(result, {"type": "genuine_question", "topic": "Billing", "specificity": "generic"})

    def test_unknown_values_fall_back_safely(self):
        result = llm.normalise_classification(
            {"type": "mystery", "topic": "Not A Topic", "specificity": "kinda"},
            ["Billing"])
        self.assertEqual(result, {"type": "unclear", "topic": "Other", "specificity": "not_applicable"})

    def test_specificity_forced_na_for_non_questions(self):
        result = llm.normalise_classification(
            {"type": "acknowledgment", "topic": "Billing", "specificity": "generic"},
            ["Billing"])
        self.assertEqual(result["specificity"], "not_applicable")

    def test_non_dict_input(self):
        result = llm.normalise_classification("garbage", ["Billing"])
        self.assertEqual(result["type"], "unclear")


class TestStratifiedSample(unittest.TestCase):
    def _frame(self, counts):
        rows = []
        for topic, n in counts.items():
            rows.extend({"_rough_topic": topic, "v": i} for i in range(n))
        return pd.DataFrame(rows)

    def test_small_topic_not_starved(self):
        df = self._frame({"Big": 500, "Small": 6})
        sample = scoping_logic.build_stratified_sample(df, "_rough_topic", 50, min_per_topic=8)
        self.assertEqual((sample["_rough_topic"] == "Small").sum(), 6)  # all of it
        self.assertEqual(len(sample), 50)

    def test_exact_size_honoured(self):
        df = self._frame({"A": 40, "B": 40, "C": 40})
        sample = scoping_logic.build_stratified_sample(df, "_rough_topic", 60, min_per_topic=8)
        self.assertEqual(len(sample), 60)

    def test_target_at_least_population_returns_all(self):
        df = self._frame({"A": 10})
        sample = scoping_logic.build_stratified_sample(df, "_rough_topic", 10)
        self.assertEqual(len(sample), 10)

    def test_reproducible(self):
        df = self._frame({"A": 100, "B": 100})
        s1 = scoping_logic.build_stratified_sample(df, "_rough_topic", 40)
        s2 = scoping_logic.build_stratified_sample(df, "_rough_topic", 40)
        self.assertTrue(s1.equals(s2))


class TestValidateMailbox(unittest.TestCase):
    def test_real_mailbox_accepted(self):
        df = pd.DataFrame({
            "Subject": ["VPN down", "Password"],
            "Body": ["My VPN connection keeps dropping every ten minutes, can you help?",
                     "I am locked out of my account and need a reset link please."],
        })
        detected = {"subject": "Subject", "body": "Body", "sender": None}
        ok, _ = core.validate_mailbox_dataframe(df, detected, confident=True)
        self.assertTrue(ok)

    def test_tracker_with_subject_column_rejected(self):
        # M7 regression: named columns alone must not pass a non-mailbox sheet.
        df = pd.DataFrame({
            "Subject": ["Task 1", "Task 2"],
            "Body": ["Done", "WIP"],
        })
        detected = {"subject": "Subject", "body": "Body", "sender": None}
        ok, message = core.validate_mailbox_dataframe(df, detected, confident=True)
        self.assertFalse(ok)
        self.assertIn("does not resemble", message)

    def test_empty_frame_rejected(self):
        ok, _ = core.validate_mailbox_dataframe(pd.DataFrame(), {"body": "Body"}, confident=False)
        self.assertFalse(ok)

    def test_unnamed_columns_need_message_like_text(self):
        df = pd.DataFrame({"col1": ["a", "b"], "col2": ["x", "y"]})
        detected = {"subject": "col1", "body": "col2", "sender": None}
        ok, _ = core.validate_mailbox_dataframe(df, detected, confident=False)
        self.assertFalse(ok)


class TestDetectColumns(unittest.TestCase):
    def test_standard_names(self):
        df = pd.DataFrame({"Subject": ["s"], "Body": ["b"], "From": ["f@x.co"]})
        detected, confident = core.detect_all_columns(df)
        self.assertEqual(detected["subject"], "Subject")
        self.assertEqual(detected["body"], "Body")
        self.assertTrue(confident)

    def test_body_fallback_never_picks_detected_subject(self):
        # M4 regression: with a named subject but no named body, the
        # content-based fallback must select a DIFFERENT column even when
        # the subject column contains the longest text.
        df = pd.DataFrame({
            "Subject": ["A very long subject line that is much longer than anything else here"] * 3,
            "Details": ["short", "short", "short"],
        })
        detected, _ = core.detect_all_columns(df)
        self.assertEqual(detected["subject"], "Subject")
        self.assertNotEqual(detected["body"], "Subject")

    def test_subject_and_body_never_same_column(self):
        df = pd.DataFrame({
            "Message Subject": ["Hello there, quick question about the VPN"] * 3,
            "Other": ["x", "y", "z"],
        })
        detected, _ = core.detect_all_columns(df)
        self.assertNotEqual(detected["subject"], detected["body"])


class TestSanitizeForCsv(unittest.TestCase):
    def test_formula_prefixes_escaped(self):
        # M1 regression: values beginning with = + - @ must not survive as
        # executable formulas in exported CSVs.
        df = pd.DataFrame({"topic": ["=HYPERLINK(\"http://x\")", "+SUM(A1)", "@cmd", "Normal"]})
        out = core.sanitize_for_csv(df)
        self.assertTrue(all(v.startswith("'") for v in out["topic"][:3]))
        self.assertEqual(out["topic"][3], "Normal")

    def test_numeric_columns_untouched(self):
        df = pd.DataFrame({"n": [-5, 3]})
        out = core.sanitize_for_csv(df)
        self.assertEqual(list(out["n"]), [-5, 3])


class TestRedactionFallback(unittest.TestCase):
    def test_fallback_flag_and_email_redaction(self):
        # C1 regression: when the name-detection engine is unavailable the
        # third return value MUST be False so the UI can disclose partial
        # protection -- and email/phone redaction must still work.
        original = core.get_presidio_analyzer
        core.get_presidio_analyzer = lambda: (_ for _ in ()).throw(RuntimeError("model missing"))
        try:
            df = pd.DataFrame({"Body": ["Contact jane.doe@corp.com or call +61 3 9999 8888 for help."]})
            out, mapping, full_protection = core.redact_dataframe(df, ["Body"])
            self.assertFalse(full_protection)
            redacted = out["Body_redacted"].iloc[0]
            self.assertNotIn("jane.doe@corp.com", redacted)
            self.assertIn("[EMAIL_ADDRESS_1]", redacted)
            self.assertNotIn("9999 8888", redacted)
        finally:
            core.get_presidio_analyzer = original

    def test_pseudonyms_are_consistent(self):
        original = core.get_presidio_analyzer
        core.get_presidio_analyzer = lambda: (_ for _ in ()).throw(RuntimeError("model missing"))
        try:
            df = pd.DataFrame({"Body": ["Mail a@b.co now", "Mail a@b.co again"]})
            out, mapping, _ = core.redact_dataframe(df, ["Body"])
            self.assertIn("[EMAIL_ADDRESS_1]", out["Body_redacted"].iloc[0])
            self.assertIn("[EMAIL_ADDRESS_1]", out["Body_redacted"].iloc[1])
            self.assertEqual(len(mapping), 1)
        finally:
            core.get_presidio_analyzer = original


class TestParseUpload(unittest.TestCase):
    def _xlsx_bytes(self, rows, header):
        from openpyxl import Workbook
        wb = Workbook()
        ws = wb.active
        ws.append(header)
        for row in rows:
            ws.append(row)
        buffer = io.BytesIO()
        wb.save(buffer)
        return buffer.getvalue()

    def test_csv_read_limit(self):
        csv_bytes = ("Subject,Body\n" + "\n".join(f"s{i},b{i}" for i in range(50))).encode()
        df = core.parse_upload(csv_bytes, "mail.csv", read_limit=10)
        self.assertEqual(len(df), 10)

    def test_xlsx_streaming_respects_limit(self):
        # H1 regression: xlsx reading must stop at the cap instead of
        # materialising the whole sheet.
        data = self._xlsx_bytes([[f"s{i}", f"body text {i}"] for i in range(200)], ["Subject", "Body"])
        df = core.parse_upload(data, "mail.xlsx", read_limit=25)
        self.assertEqual(len(df), 25)
        self.assertEqual(list(df.columns), ["Subject", "Body"])

    def test_xlsx_duplicate_and_blank_headers_deduped(self):
        data = self._xlsx_bytes([["a", "b", "c"]], ["Subject", "Subject", None])
        df = core.parse_upload(data, "mail.xlsx", read_limit=None)
        self.assertEqual(len(set(df.columns)), 3)


class TestRunAnalysis(unittest.TestCase):
    def test_counts_reconcile(self):
        df = pd.DataFrame({
            "Subject": ["Password help", "Thanks", "VPN question", "Password again"],
            "Body": ["How do I reset my password please?",
                     "Thanks, noted.",
                     "How does the vpn certificate renewal work?",
                     "I am locked out, can you reset my password?"],
            "Sender": ["a", "b", "c", "d"],
        })
        analyzed, backlog, type_summary = core.run_analysis(
            df, core.DEFAULT_TOPICS, "Subject", "Body", "Sender")
        self.assertEqual(int(type_summary["count"].sum()), len(df))
        pw_row = backlog[backlog["topic"] == "Password Reset"].iloc[0]
        self.assertEqual(int(pw_row["genuine_questions"]), 2)
        self.assertEqual(int(backlog["rank"].iloc[0]), 1)


class TestCostEstimates(unittest.TestCase):
    def test_zero_emails_zero_cost(self):
        cost, seconds = scoping_logic.estimate_time_and_cost(0, 15, "claude-haiku-4-5-20251001")
        self.assertEqual((cost, seconds), (0.0, 0))

    def test_sonnet_costs_more_than_haiku(self):
        haiku = llm.estimate_cost(1000, "claude-haiku-4-5-20251001")
        sonnet = llm.estimate_cost(1000, "claude-sonnet-4-6")
        self.assertGreater(sonnet, haiku)


if __name__ == "__main__":
    unittest.main(verbosity=2)
