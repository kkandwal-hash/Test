import argparse
import csv
import json
import os
import sys
import time
from typing import Dict

from openai import OpenAI


ALLOWED_WORKFLOWS = {"Project", "Asset", "Approval", "Record", "Integration"}
ALLOWED_INTENTS = {"How-to", "Bug", "Feature Request", "Feedback"}
ALLOWED_SENTIMENTS = {"Positive", "Neutral", "Negative"}


def build_prompt(user_question: str) -> str:
	return f"""
	You are an annotation assistant.
	Tag the user question with the following schema:

	Workflow: [Project | Asset | Approval | Record | Integration]
	Intent: [How-to | Bug | Feature Request | Feedback]
	Sentiment: [Positive | Neutral | Negative]

	Question: "{user_question}"
	Provide output ONLY in valid JSON.
	Example: {{"Workflow": "Asset", "Intent": "How-to", "Sentiment": "Neutral"}}
	"""


def sanitize_tags(raw: Dict[str, str]) -> Dict[str, str]:
	workflow = raw.get("Workflow") or raw.get("workflow") or "Unknown"
	intent = raw.get("Intent") or raw.get("intent") or "Unknown"
	sentiment = raw.get("Sentiment") or raw.get("sentiment") or "Unknown"

	workflow = workflow if workflow in ALLOWED_WORKFLOWS else "Unknown"
	intent = intent if intent in ALLOWED_INTENTS else "Unknown"
	sentiment = sentiment if sentiment in ALLOWED_SENTIMENTS else "Unknown"

	return {"Workflow": workflow, "Intent": intent, "Sentiment": sentiment}


def tag_question(
	client: OpenAI,
	question: str,
	model: str,
	max_retries: int = 3,
	sleep_between_retries_sec: float = 1.0,
) -> Dict[str, str]:
	prompt = build_prompt(question)
	last_error: Exception | None = None

	for attempt in range(max_retries):
		try:
			response = client.chat.completions.create(
				model=model,
				messages=[
					{"role": "system", "content": "You are an annotation assistant."},
					{"role": "user", "content": prompt},
				],
				response_format={"type": "json_object"},
				temperature=0,
			)
			content = response.choices[0].message.content or "{}"
			parsed = json.loads(content)
			return sanitize_tags(parsed)
		except Exception as exc:  # Broad catch to retry on API/parse/network errors
			last_error = exc
			time.sleep(sleep_between_retries_sec * (2 ** attempt))

	# Fallback if we exhausted retries
	return {"Workflow": "Unknown", "Intent": "Unknown", "Sentiment": "Unknown"}


def process_csv(
	input_path: str,
	output_path: str,
	model: str,
	sleep_between_requests_sec: float,
	max_retries: int,
	max_rows: int | None,
) -> None:
	api_key = os.getenv("OPENAI_API_KEY")
	if not api_key:
		raise RuntimeError("OPENAI_API_KEY is not set in the environment.")

	client = OpenAI()

	processed = 0

	with open(input_path, "r", encoding="utf-8", newline="") as infile, open(
		output_path, "w", encoding="utf-8", newline=""
	) as outfile:
		reader = csv.DictReader(infile)
		fieldnames = ["id", "question", "workflow", "intent", "sentiment"]
		writer = csv.DictWriter(outfile, fieldnames=fieldnames)
		writer.writeheader()

		for row in reader:
			question = row.get("question", "").strip()
			row_id = row.get("id", "").strip()

			if not question:
				writer.writerow(
					{
						"id": row_id,
						"question": question,
						"workflow": "Unknown",
						"intent": "Unknown",
						"sentiment": "Unknown",
					}
				)
				processed += 1
				continue

			tags = tag_question(
				client=client,
				question=question,
				model=model,
				max_retries=max_retries,
			)

			writer.writerow(
				{
					"id": row_id,
					"question": question,
					"workflow": tags.get("Workflow", "Unknown"),
					"intent": tags.get("Intent", "Unknown"),
					"sentiment": tags.get("Sentiment", "Unknown"),
				}
			)
			outfile.flush()

			processed += 1
			if processed % 25 == 0:
				print(f"Tagged {processed} rows...", flush=True)

			if max_rows is not None and processed >= max_rows:
				break

			time.sleep(sleep_between_requests_sec)

	print(f"✅ Tagging complete. Processed {processed} rows. Results saved to {output_path}")


def parse_args(argv: list[str]) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Tag questions using OpenAI models")
	parser.add_argument("--input", "-i", default="questions.csv", help="Input CSV path")
	parser.add_argument(
		"--output", "-o", default="tagged_questions.csv", help="Output CSV path"
	)
	parser.add_argument(
		"--model", "-m", default="gpt-4o-mini", help="Model name (e.g., gpt-4o-mini)"
	)
	parser.add_argument(
		"--sleep", type=float, default=0.0, help="Sleep seconds between requests"
	)
	parser.add_argument(
		"--retries", type=int, default=3, help="Max retries per request"
	)
	parser.add_argument(
		"--max-rows", type=int, default=None, help="Process at most N rows (for testing)"
	)
	return parser.parse_args(argv)


def main() -> None:
	args = parse_args(sys.argv[1:])
	process_csv(
		input_path=args.input,
		output_path=args.output,
		model=args.model,
		sleep_between_requests_sec=args.sleep,
		max_retries=args.retries,
		max_rows=args.max_rows,
	)


if __name__ == "__main__":
	main()

