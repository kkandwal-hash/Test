import re
import string
import sys
import json

try:
    import pandas as pd  # Optional; falls back to csv module if unavailable
except Exception:
    pd = None

_STOPWORDS = {
    "a","an","the","and","or","but","if","then","else","for","to","of","in","on","at","by","with","as","is","are","was","were",
    "be","being","been","it","its","this","that","these","those","i","you","he","she","we","they","them","me","my","your","our",
    "us","do","does","did","can","could","should","would","will","shall","may","might","not"
}

_LIST_MARKERS_RE = re.compile(r'^\s*(?:\d+\.\s+|[-*]\s+)', re.MULTILINE)
_BULLET_CHARS = {"-", "*", "•"}
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def _normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


def _tokens(text: str) -> set:
    if not text:
        return set()
    t = text.lower().translate(_PUNCT_TABLE)
    toks = [w for w in t.split() if len(w) >= 3 and w not in _STOPWORDS]
    return set(toks)


def auto_tag(message, answer, time_took):
    tags = []

    message = "" if not isinstance(message, str) else message
    answer = "" if not isinstance(answer, str) else answer

    msg = _normalize(message)
    ans = _normalize(answer)

    # Accuracy
    if re.search(r"\b(fake|fabricated|made[- ]?up|not real)\b", ans):
        tags.append("Accuracy: Hallucination")
    if any(phrase in ans for phrase in ["not available", "unknown", "no data", "i don't know"]) \
       or (len(ans.split()) < 25 and any(q in msg for q in ["how ", "what ", "why ", "where ", "when ", "which "])):
        tags.append("Accuracy: Missing Data")

    # Completeness
    if any(phrase in ans for phrase in ["beta", "cannot provide", "limited information", "insufficient data"]):
        tags.append("Completeness: Incomplete")

    # Clarity
    if "..." in answer or (answer.count("\n") == 0 and len(ans.split()) > 80):
        tags.append("Clarity: Confusing")

    # Contextual Fit (Jaccard-like after stopword/punct removal)
    msg_tokens = _tokens(message)
    ans_tokens = _tokens(answer)
    if ans_tokens:
        intersection = len(msg_tokens & ans_tokens)
        union = len(msg_tokens | ans_tokens) or 1
        jaccard = intersection / union
        if jaccard < 0.03 and len(ans) > 0:
            tags.append("Contextual Fit: Off-topic")

    # Actionability
    ask_action = any(k in msg for k in ["create", "update", "summarize", "write", "generate", "fix", "steps", "instructions"])
    has_action = any(k in ans for k in ["create", "update", "summarize", "next steps", "step", "do this", "you can"])
    has_list = bool(_LIST_MARKERS_RE.search(answer))
    if ask_action and not (has_action or has_list):
        tags.append("Actionability: Not actionable")

    # Format
    if "summarize" in msg and not (has_list or any(c in answer for c in _BULLET_CHARS)):
        tags.append("Format: Wrong format")

    # Performance
    try:
        took = float(time_took)
    except Exception:
        took = None
    if took is not None and took > 10:
        tags.append("Performance: Too slow")
    if len(ans) == 0 or any(w in ans for w in ["error", "exception", "traceback"]):
        tags.append("Performance: Broken response")

    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for t in tags:
        if t not in seen:
            seen.add(t)
            deduped.append(t)
    return deduped


def main():
    input_path = "/workspace/questions.csv"
    output_path = "/workspace/questions_with_auto_tags.csv"
    if pd is not None:
        df = pd.read_csv(input_path)
        df["auto_tags"] = df.apply(
            lambda row: auto_tag(row.get("message"), row.get("answer"), row.get("time_took")),
            axis=1,
        ).apply(json.dumps)
        df.to_csv(output_path, index=False)
    else:
        import csv

        with open(input_path, newline="", encoding="utf-8") as f_in:
            reader = csv.DictReader(f_in)
            fieldnames = list(reader.fieldnames or [])
            if "auto_tags" not in fieldnames:
                fieldnames.append("auto_tags")
            rows = []
            for row in reader:
                message = row.get("message")
                answer = row.get("answer")
                time_took = row.get("time_took")
                tags = auto_tag(message, answer, time_took)
                row["auto_tags"] = json.dumps(tags)
                rows.append(row)

        with open(output_path, "w", newline="", encoding="utf-8") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(f"✅ File saved: {output_path}")


if __name__ == "__main__":
    sys.exit(main())

