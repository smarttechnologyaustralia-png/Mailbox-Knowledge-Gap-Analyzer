"""
LLM-powered classification -- an alternative to the rule-based classifier,
built specifically to solve the "any dataset" problem the regex approach
can't: it doesn't need pre-defined keywords, doesn't care what language or
industry the mailbox is from, and can actually judge whether a question is
generic or needs personal case lookup (the thing the rule-based version
got wrong on the Printer/Hardware topic).

Requires: pip install anthropic
Requires: an Anthropic API key (never hardcode it -- always pass it in
at runtime, e.g. via a password-masked input field).

HONESTY NOTE: this file was written and its JSON-parsing logic was tested
against simulated (mocked) API responses -- the actual live API call
could not be tested in the sandbox this was built in, since it has no
internet access. Test the real call yourself before relying on it.
"""
import json
import re

BATCH_SIZE = 15  # emails per API call -- balances cost/speed against
                 # keeping each prompt small enough to get a reliable,
                 # complete JSON response back.


def _get_response_text(response):
    """Robustly pulls the text out of a Claude API response. Some
    responses include a 'thinking' block before the actual text block --
    content[0] is not always safe to assume is text. This checks every
    block and returns the first one that actually has text, instead of
    blindly grabbing the first block regardless of type."""
    for block in response.content:
        if hasattr(block, "text"):
            return block.text
    raise ValueError("No text block found in the API response — got block types: "
                      f"{[type(b).__name__ for b in response.content]}")


def _extract_json(text):
    """LLMs sometimes wrap JSON in ```json fences or add a sentence before
    it despite instructions not to. This pulls out just the JSON part."""
    text = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)
    else:
        bracket_match = re.search(r"(\[.*\])", text, re.DOTALL)
        if bracket_match:
            text = bracket_match.group(1)
    return json.loads(text)


def discover_topics_with_llm(client, model, sample_emails, max_topics=10):
    """
    sample_emails: list of dicts with 'subject' and 'body' keys.
    Returns a list of topic name strings, discovered fresh from the data --
    no pre-defined taxonomy needed. This is what lets the tool work on a
    completely unfamiliar business domain.
    """
    numbered = "\n\n".join(
        f"Email {i+1}:\nSubject: {e['subject']}\nBody: {e['body'][:300]}"
        for i, e in enumerate(sample_emails)
    )
    prompt = f"""Here are {len(sample_emails)} sample emails from a business support mailbox.

{numbered}

Based on these examples, identify up to {max_topics} recurring topic categories
that would sensibly organize this mailbox (e.g. "Password Reset", "Billing
Question", "Order Status" -- whatever actually fits THESE emails, not a
generic guess).

Do not create a category for pure acknowledgments, automated notifications,
or one-off unrelated messages.

Return ONLY a JSON array of topic name strings, nothing else. Example format:
["Topic One", "Topic Two", "Topic Three"]"""

    response = client.messages.create(
        model=model, max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    raw_text = _get_response_text(response)
    topics = _extract_json(raw_text)
    return [str(t) for t in topics]


def classify_batch_with_llm(client, model, email_batch, topics):
    """
    email_batch: list of dicts with 'subject' and 'body' keys.
    topics: list of topic name strings (from discover_topics_with_llm,
            or user-supplied).
    Returns a list of dicts, one per email, in the SAME ORDER as the input:
        {"type": "genuine_question" | "acknowledgment" | "status_update"
                 | "automated" | "unclear",
         "topic": one of the given topics, or "Other",
         "specificity": "generic" | "case_specific" | "not_applicable"}
    """
    numbered = "\n\n".join(
        f"Email {i+1}:\nSubject: {e['subject']}\nBody: {e['body'][:400]}"
        for i, e in enumerate(email_batch)
    )
    topic_list = ", ".join(f'"{t}"' for t in topics)

    prompt = f"""Classify each of the following {len(email_batch)} emails for a
self-service knowledge gap analysis.

Available topics: {topic_list}, or "Other" if none fit.

For each email, determine:
1. type: "genuine_question" (sender is asking something that needs an
   answer), "acknowledgment" (thanks/confirmation, nothing being asked),
   "status_update" (sharing information, no question), "automated" (system
   notification, not a real person writing), or "unclear" (empty or
   ambiguous).
2. topic: the single best-fitting topic from the list above, or "Other".
3. specificity: only fill this in if type is "genuine_question" --
   "generic" if a general help article could answer this for anyone who
   asks it, "case_specific" if it requires looking up this person's
   individual situation (a specific order, account, or personal
   circumstance) and a generic article could not answer it. Use
   "not_applicable" if type is not genuine_question.

{numbered}

Return ONLY a JSON array with exactly {len(email_batch)} objects, in the
same order as the emails above. Example format:
[{{"type": "genuine_question", "topic": "Billing", "specificity": "generic"}}, ...]"""

    response = client.messages.create(
        model=model, max_tokens=200 * len(email_batch),
        messages=[{"role": "user", "content": prompt}],
    )
    raw_text = _get_response_text(response)
    results = _extract_json(raw_text)

    if len(results) != len(email_batch):
        raise ValueError(
            f"Expected {len(email_batch)} classifications back, got {len(results)}. "
            f"The model's response may have been truncated -- try a smaller batch size."
        )
    return results


def estimate_cost(num_emails, model="claude-haiku-4-5-20251001"):
    """Rough order-of-magnitude cost estimate, NOT a quote. Assumes ~400
    input tokens and ~60 output tokens per email once batching overhead is
    spread out, at approximate Haiku-tier pricing. Actual cost depends on
    real email length and current pricing -- check your provider's
    current rates before relying on this number."""
    est_input_tokens = num_emails * 400
    est_output_tokens = num_emails * 60
    # Approximate standard API rates in USD/MTok. This remains an estimate:
    # actual usage depends on message length and provider pricing.
    if "sonnet" in model.lower():
        input_rate_per_million, output_rate_per_million = 3.0, 15.0
    else:
        input_rate_per_million, output_rate_per_million = 1.0, 5.0
    cost = (est_input_tokens / 1_000_000 * input_rate_per_million +
            est_output_tokens / 1_000_000 * output_rate_per_million)
    return round(cost, 2)
