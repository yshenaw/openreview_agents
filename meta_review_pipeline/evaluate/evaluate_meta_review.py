#!/usr/bin/env python3
"""Run the meta-review evaluation for a single submission folder."""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Tuple

from ..utils.api_client import AzureOpenAIClient, OpenAIClient
from ..utils.export_process import read_text_file


def _candidate_review_dirs(submission_folder: str) -> List[str]:
    """Return directories that may hold review or meta-review files."""

    candidates: List[str] = []
    for dirname in ("reviews", "Reviews"):
        path = os.path.join(submission_folder, dirname)
        if os.path.isdir(path):
            candidates.append(path)

    candidates.append(submission_folder)

    unique: List[str] = []
    seen: set[str] = set()
    for path in candidates:
        canonical = os.path.abspath(path)
        if canonical not in seen:
            unique.append(path)
            seen.add(canonical)
    return unique


def _collect_reviews(submission_folder: str) -> Dict[str, str]:
    """Load all review_*.txt files as a mapping {review_id: content}."""

    reviews: Dict[str, str] = {}
    for directory in _candidate_review_dirs(submission_folder):
        if not os.path.isdir(directory):
            continue
        for filename in sorted(os.listdir(directory)):
            lower = filename.lower()
            if lower.startswith("review_") and lower.endswith(".txt"):
                review_id = filename[len("review_") : -len(".txt")]
                if review_id in reviews:
                    continue
                path = os.path.join(directory, filename)
                reviews[review_id] = read_text_file(path)
    return reviews


def _pick_meta_review(submission_folder: str, meta_review_file: Optional[str]) -> Tuple[str, str]:
    """Return (meta_review_text, file_path) for the desired meta-review."""

    if meta_review_file:
        candidate = meta_review_file
        if not os.path.isabs(candidate):
            located: Optional[str] = None
            for directory in _candidate_review_dirs(submission_folder):
                path = os.path.join(directory, candidate)
                if os.path.exists(path):
                    located = path
                    break
            candidate = located or os.path.join(submission_folder, candidate)
        if not os.path.exists(candidate):
            raise FileNotFoundError(f"Specified meta-review file not found: {candidate}")
        return read_text_file(candidate), candidate

    for directory in _candidate_review_dirs(submission_folder):
        priority_lists = [
            ["meta_review.txt"],
            sorted(
                f
                for f in os.listdir(directory)
                if f.lower().startswith("meta_review") and f.lower().endswith(".txt")
            ),
            sorted(
                f
                for f in os.listdir(directory)
                if f.lower().startswith("generated_meta_review") and f.lower().endswith(".txt")
            ),
        ]
        for candidates in priority_lists:
            for candidate in candidates:
                path = os.path.join(directory, candidate)
                if os.path.exists(path):
                    return read_text_file(path), path

    raise FileNotFoundError(
        "No meta-review file found. Specify one with --meta-review-file or place a meta_review*.txt file in the submission folder."
    )


def _load_confidential_note(submission_folder: str) -> str:
    """Load submission_discussion.txt if present."""

    path = os.path.join(submission_folder, "submission_discussion.txt")
    if os.path.exists(path):
        return read_text_file(path)
    return ""


def normalize_evaluation_decision(raw_decision: str) -> str:
    """Normalize evaluation decision token into REWRITE, OK, or UNKNOWN."""

    decision = (raw_decision or "").strip().upper()
    token = "".join(ch for ch in decision if ch.isalpha())
    if token.startswith("REWRITE") or "REWRITE" in token or "REVISE" in token or "REVISION" in token:
        return "REWRITE"
    if "OK" in token or "KEEP" in token or "ACCEPTABLE" in token:
        return "OK"
    return "UNKNOWN"


def normalize_conflict_flag(flag: str) -> str:
    """Normalize conflict flag into YES, NO, or UNKNOWN."""

    token = (flag or "").strip().upper()
    clean = "".join(ch for ch in token if ch.isalpha())
    if clean.startswith("YES") or clean == "YES":
        return "YES"
    if clean.startswith("NO") or clean == "NO":
        return "NO"
    return "UNKNOWN"


def extract_rewrite_reason(raw_decision: str) -> str:
    """Extract a short rewrite reason from the raw decision string."""

    if not raw_decision:
        return ""

    parts = raw_decision.split(":", 1)
    if len(parts) == 2 and "REWRITE" in parts[0].upper():
        return parts[1].strip()

    upper = raw_decision.upper()
    if "REWRITE" in upper:
        keyword_index = upper.find("REWRITE")
        fragment = raw_decision[keyword_index + len("REWRITE") :].strip(" :-")
        if fragment:
            first_sentence = fragment.split(".", 1)[0].strip()
            return first_sentence or fragment
    return ""


def interpret_evaluation_response(raw_response: str) -> Dict[str, str]:
    """Parse the model response into normalized fields."""

    decision_token = ""
    rewrite_reason_token = ""
    conflict_token = ""
    conflict_reason_token = ""

    for line in (raw_response or "").splitlines():
        stripped = line.strip()
        upper = stripped.upper()

        if upper.startswith("REWRITE_DECISION"):
            _, _, remainder = stripped.partition(":")
            value = remainder.strip()
            token, _, reason = value.partition("-")
            decision_token = token.strip()
            rewrite_reason_token = reason.strip()
        elif upper.startswith("CONFLICT_WITH_REVIEWS"):
            _, _, remainder = stripped.partition(":")
            value = remainder.strip()
            token, _, reason = value.partition("-")
            conflict_token = token.strip()
            conflict_reason_token = reason.strip()

    normalized_decision = normalize_evaluation_decision(decision_token)
    normalized_conflict = normalize_conflict_flag(conflict_token)

    rewrite_reason = rewrite_reason_token
    if normalized_decision != "REWRITE":
        rewrite_reason = ""
    elif not rewrite_reason:
        rewrite_reason = extract_rewrite_reason(decision_token)

    return {
        "decision": normalized_decision,
        "rewrite_reason": rewrite_reason,
        "conflict": normalized_conflict,
        "conflict_reason": conflict_reason_token,
    }


def evaluate_meta_review(
    client,
    reviews: Dict[str, str],
    meta_review: str,
    confidential_note: str = "",
    score_statement: str = "",
    api_provider: str = "azure",
) -> str:
    """Run the meta-review evaluation prompt using the provided LLM client.

    Parameters
    ----------
    client: AzureOpenAIClient | OpenAIClient
        Configured Azure OpenAI or OpenAI client.
    reviews: Dict[str, str]
        Mapping of review identifiers to their text content.
    meta_review: str
        The meta-review text under evaluation.
    confidential_note: str, optional
        Author-to-AC confidential note, if available.
    score_statement: str, optional
        Description of the reviewer scoring scale to provide additional context.
    """

    prompt = (
        "You are an expert Senior Area Chair (SAC) for ICLR.\n"
        "Your goal is to identify questionable meta-reviews that need revision before sending to authors"
        " and ensure that the meta-reviews are of high quality.\n"
        "Good meta-reviews should clearly justify the final decision, summarize the overall reviewer sentiment, and explain how the authors’ feedback was taken into account. One or of each point is sufficient and should be flagged as <OK> in REWRITE_DECISION.\n"
        "Please evaluate whether the current meta-review should be rewritten before sharing with authors.\n"
        "Do not make a publication decision about the paper itself.\n"
        "Also determine whether the meta-review conflicts with the reviewers' overall stance."
        "Maintain high standards for publications. If the meta-review is negative and at least one individual review is also negative, this situation alone should not be considered a conflict. If you still believe it constitutes a conflict, ensure that the primary concerns raised in the reviews have been adequately addressed by the authors.\n"
        "Respond exactly with two lines in this format:\n"
        "REWRITE_DECISION: <REWRITE or OK> - <one-sentence for which point is missing (use 'OK' with no reason if appropriate)>\n"
        "CONFLICT_WITH_REVIEWS: <YES or NO> - <one-sentence explanation>."
    )

    reviews_text = "\n\n=== REVIEWER COMMENTS ===\n"
    for key, value in reviews.items():
        reviews_text += f"\n--- Review {key} ---\n{value}\n"

    meta_review_block = f"\n\n=== META-REVIEW ===\n{meta_review}\n"

    score_block = ""
    if score_statement:
        score_block = (
            "\n\n=== REVIEW SCORING GUIDELINE ===\n"
            f"{score_statement.strip()}\n"
        )

    note_block = ""
    if confidential_note and confidential_note.strip():
        note_block = (
            "\n\n=== CONFIDENTIAL AUTHOR → AC NOTE (Subjective context; use cautiously) ===\n"
            f"{confidential_note.strip()}\n"
        )

    evaluation_prompt = (
        prompt
        + reviews_text
        + meta_review_block
        + score_block
        + note_block
        + "\nReply strictly using the two-line format described above."
    )

    print(f"Running meta-review rewrite check ({api_provider})...")

    if api_provider == "azure":
        response_text = client.chat_completion(
            model=client.evaluation_deployment,
            messages=[{"role": "user", "content": [{"type": "text", "text": evaluation_prompt}]}],
            temperature=0.0,
        )
    elif api_provider in {"openai", "openai-url", "openai_url", "pdf-url"}:
        response_text = client.evaluate_with_text(
            prompt_text=evaluation_prompt,
            temperature=0.0,
        )
    else:
        raise ValueError(f"Unsupported evaluation API: {api_provider}")

    return response_text.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run meta-review evaluation on a submission folder")
    parser.add_argument("submission_folder", help="Path to SubmissionXXXX folder")
    parser.add_argument(
        "--meta-review-file",
        help="Specific meta-review filename to evaluate (relative to the submission folder)",
    )
    parser.add_argument(
        "--output",
        help="Optional path to write the evaluation decision and raw response",
    )
    parser.add_argument(
        "--score-statement",
        help="Optional textual description of the reviewer scoring scale to include in the prompt",
    )
    parser.add_argument(
        "--api",
        choices=["azure", "openai"],
        default="azure",
        help="LLM provider to use for evaluation (default: azure)",
    )
    args = parser.parse_args()

    submission_folder = args.submission_folder
    if not os.path.isdir(submission_folder):
        raise FileNotFoundError(f"Submission folder not found: {submission_folder}")

    reviews = _collect_reviews(submission_folder)
    if not reviews:
        raise RuntimeError("No review_*.txt files found in the submission folder.")

    meta_review_text, meta_review_path = _pick_meta_review(submission_folder, args.meta_review_file)
    confidential_note = _load_confidential_note(submission_folder)

    api_normalized = args.api.lower()
    if api_normalized == "azure":
        client = AzureOpenAIClient()
    elif api_normalized in {"openai", "openai-url", "openai_url", "pdf-url"}:
        client = OpenAIClient()
    else:
        raise ValueError(f"Unsupported evaluation API: {args.api}")

    print(f"Running meta-review rewrite check using {api_normalized} API...")
    print(f"Submission folder: {submission_folder}")
    print(f"Meta-review file: {meta_review_path}")
    if confidential_note:
        print("Found confidential author note (submission_discussion.txt).")
    else:
        print("No submission_discussion.txt found.")

    raw_response = evaluate_meta_review(
        client=client,
        reviews=reviews,
        meta_review=meta_review_text,
        confidential_note=confidential_note,
        score_statement=args.score_statement or "",
        api_provider=api_normalized,
    )

    interpreted = interpret_evaluation_response(raw_response)
    normalized_decision = interpreted["decision"]
    rewrite_reason = interpreted["rewrite_reason"]
    conflict_flag = interpreted["conflict"]
    conflict_reason = interpreted["conflict_reason"]

    print("\nMeta-review rewrite assessment:")
    print(f"  Normalized: {normalized_decision}")
    if rewrite_reason:
        print(f"  Reason: {rewrite_reason}")
    print(f"  Conflict with reviews: {conflict_flag}")
    if conflict_reason:
        print(f"    Explanation: {conflict_reason}")
    print(f"  Raw model output: {raw_response}")

    if args.output:
        output_path = args.output
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(f"Submission: {submission_folder}\n")
            handle.write(f"Meta-review file: {meta_review_path}\n")
            handle.write(f"Assessment: {normalized_decision}\n")
            if rewrite_reason:
                handle.write(f"Reason: {rewrite_reason}\n")
            handle.write(f"Conflict with reviews: {conflict_flag}\n")
            if conflict_reason:
                handle.write(f"Conflict explanation: {conflict_reason}\n")
            handle.write("Raw Output:\n")
            handle.write(raw_response if raw_response.endswith("\n") else raw_response + "\n")
    print(f"\nWrote meta-review evaluation to {output_path}")


if __name__ == "__main__":
    main()