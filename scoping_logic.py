"""
Scoping logic: lets the user choose how much of the mailbox to actually
send to the AI, with real cost/time numbers, before anything expensive
runs. Uses a free, instant keyword pre-scan (no API call) to estimate
per-topic volume across the FULL dataset, then builds a stratified sample
so small topics aren't starved out if the user chooses to sample rather
than analyze everything.
"""
import math
import re

import pandas as pd


def quick_keyword_prescan(df, subject_col, body_col, topics):
    """
    Free, instant, works on any size dataset. Not the real classification --
    just a rough sizing tool so cost/time estimates and sampling can be
    based on real proportions instead of a blind guess. Each email gets
    assigned to whichever topic's name-words appear most, or 'Other'.
    """
    text = (df[subject_col].fillna("").astype(str) + " " + df[body_col].fillna("").astype(str)).str.lower()

    def best_topic(t):
        best, best_count = "Other", 0
        for topic in topics:
            words = [w for w in re.findall(r"[a-z]+", topic.lower()) if len(w) > 2]
            count = sum(t.count(w) for w in words)
            if count > best_count:
                best, best_count = topic, count
        return best

    return text.apply(best_topic)


def estimate_time_and_cost(n_emails, batch_size, model, seconds_per_call=6):
    """Rough estimate, not a guarantee -- real latency varies with network
    conditions and response length. n_emails=0 returns zero for both."""
    from llm_classifier import estimate_cost, estimate_article_drafting_cost
    if n_emails == 0:
        return 0.0, 0
    n_batches = math.ceil(n_emails / batch_size)
    n_calls = n_batches + 1  # +1 for the final article-pack drafting call
    est_seconds = n_calls * seconds_per_call
    est_cost = estimate_cost(n_emails, model) + estimate_article_drafting_cost(model)
    return est_cost, est_seconds


def format_time(seconds):
    if seconds < 60:
        return f"~{seconds}s"
    minutes = seconds / 60
    return f"~{minutes:.1f} min"


def build_stratified_sample(df, rough_topic_col, target_total, min_per_topic=8, seed=42):
    """
    Builds a sample of size <= target_total that covers every topic fairly,
    rather than a plain random sample that could accidentally miss a small
    topic entirely. Each topic gets at least min_per_topic examples (or all
    of it, if it has fewer than that), then remaining budget is spread
    proportionally to each topic's real share of the population.
    """
    total_pop = len(df)
    if target_total >= total_pop:
        return df.copy()

    topic_counts = df[rough_topic_col].value_counts()
    topics = topic_counts.index.tolist()

    # Step 1: floor allocation
    floor_alloc = {t: min(min_per_topic, topic_counts[t]) for t in topics}
    floor_total = sum(floor_alloc.values())

    if floor_total >= target_total:
        # Even the floors don't fit -- scale them down proportionally.
        scale = target_total / floor_total
        alloc = {t: max(1, int(round(v * scale))) for t, v in floor_alloc.items()}
    else:
        # Step 2: distribute the remaining budget proportionally to each
        # topic's share of what's left after the floor is set aside.
        remaining_budget = target_total - floor_total
        remaining_pop = {t: topic_counts[t] - floor_alloc[t] for t in topics}
        remaining_pop_total = sum(remaining_pop.values())
        alloc = dict(floor_alloc)
        if remaining_pop_total > 0:
            for t in topics:
                share = remaining_pop[t] / remaining_pop_total
                extra = int(round(share * remaining_budget))
                alloc[t] += min(extra, remaining_pop[t])

    # Correct rounding so the requested size is honoured exactly whenever
    # the population has enough rows.
    while sum(alloc.values()) > target_total:
        reducible = [t for t in topics if alloc[t] > 1]
        if not reducible:
            break
        t = max(reducible, key=lambda name: alloc[name])
        alloc[t] -= 1
    while sum(alloc.values()) < target_total:
        expandable = [t for t in topics if alloc[t] < topic_counts[t]]
        if not expandable:
            break
        t = max(expandable, key=lambda name: topic_counts[name] - alloc[name])
        alloc[t] += 1

    # Step 3: actually sample the rows
    sampled_parts = []
    for t, n in alloc.items():
        pool = df[df[rough_topic_col] == t]
        n = min(n, len(pool))
        if n > 0:
            sampled_parts.append(pool.sample(n=n, random_state=seed))
    result = pd.concat(sampled_parts) if sampled_parts else df.iloc[0:0]
    return result
