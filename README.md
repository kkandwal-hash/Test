# wf-ai-feedback

## Tag questions

Prereqs:
- Python 3.9+
- Environment variable `OPENAI_API_KEY` set

Install deps:

```bash
pip install -r requirements.txt
```

Run tagging:

```bash
python tag_questions.py --input questions.csv --output tagged_questions.csv --model gpt-4o-mini
```

Flags:
- `--sleep` seconds between requests (default 0)
- `--retries` per request (default 3)
- `--max-rows` limit for testing
