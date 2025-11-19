#!/usr/bin/env python3
"""Batch meta-review evaluation for exported submissions."""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from ..utils.api_client import AzureOpenAIClient, OpenAIClient
from ..utils.export_process import read_text_file
from .evaluate_meta_review import (
    _candidate_review_dirs,
    _collect_reviews,
    _load_confidential_note,
    evaluate_meta_review,
    interpret_evaluation_response,
)


def _find_meta_review(
    submission_folder: str,
    preferred_mode: Optional[str],
) -> Tuple[str, str]:
    """Locate a meta-review file and return (text, path)."""

    submission_id = os.path.basename(submission_folder.rstrip(os.sep))

    for directory in _candidate_review_dirs(submission_folder):
        priority_lists = [
            ["meta_review.txt"],
            sorted(
                f
                for f in os.listdir(directory)
                if f.lower().startswith("meta_review") and f.lower().endswith(".txt")
            ),
            _prioritized_generated_meta_reviews(directory, preferred_mode),
        ]

        for candidates in priority_lists:
            for candidate in candidates:
                path = os.path.join(directory, candidate)
                if os.path.exists(path):
                    return read_text_file(path), path

    raise FileNotFoundError(
        f"No meta-review file found for {submission_id}. Place meta_review*.txt files inside the submission folder."
    )


def _prioritized_generated_meta_reviews(directory: str, preferred_mode: Optional[str]) -> List[str]:
    """Return generated meta-review filenames ordered by preferred mode when available."""

    candidates = [
        f
        for f in os.listdir(directory)
        if "generated_meta_review" in f.lower() and f.lower().endswith(".txt")
    ]

    if not candidates:
        return []

    if preferred_mode:
        preferred_lower = preferred_mode.lower()
        candidates.sort(key=lambda name: (preferred_lower not in name.lower(), name))
    else:
        candidates.sort()

    return candidates


def _parse_title_line(line: str) -> Optional[str]:
    """Attempt to parse a paper title from a metadata line."""

    stripped = (line or "").strip()
    if not stripped:
        return None

    lowered = stripped.lower()
    if lowered.startswith("paper title:"):
        _, _, remainder = stripped.partition(":")
        candidate = remainder.strip()
        return candidate or None

    if lowered.startswith("title:"):
        _, _, remainder = stripped.partition(":")
        candidate = remainder.strip()
        if candidate and candidate.lower() not in {"paper decision"}:
            return candidate
        return None

    if lowered.startswith("paper #"):
        _, _, remainder = stripped.partition(":")
        candidate = remainder.strip()
        return candidate or None

    return None


def _extract_paper_title(
    submission_folder: str,
    meta_review_text: str,
    reviews: Dict[str, str],
) -> Optional[str]:
    """Derive the paper title from meta-review content, reviews, or file names."""

    for line in (meta_review_text or "").splitlines():
        title = _parse_title_line(line)
        if title:
            return title

    for review_text in reviews.values():
        for line in review_text.splitlines():
            title = _parse_title_line(line)
            if title:
                return title

    pdf_candidates = [
        f
        for f in os.listdir(submission_folder)
        if f.lower().endswith(".pdf")
    ]
    if pdf_candidates:
        pdf_stem = os.path.splitext(pdf_candidates[0])[0]
        parts = pdf_stem.split("_", 1)
        if len(parts) == 2 and parts[1]:
            return parts[1].replace("_", " ").strip()
        return pdf_stem.replace("_", " ").strip()

    return None


def batch_evaluate_meta_reviews(
    forum_folder: str,
    target_submission_folder: Optional[str],
    output_folder: str,
    preferred_mode: Optional[str],
    score_statement: str = "",
    api_provider: str = "azure",
) -> None:
    """Evaluate meta-reviews for every SubmissionXXXX folder under `forum_folder`."""

    os.makedirs(output_folder, exist_ok=True)

    if target_submission_folder:
        target_submission_folder = os.path.abspath(target_submission_folder)
        if not os.path.isdir(target_submission_folder):
            raise FileNotFoundError(f"Submission folder not found: {target_submission_folder}")

        base_name = os.path.basename(target_submission_folder.rstrip(os.sep)).lower()
        if "submission" in base_name:
            submissions = [target_submission_folder]
        else:
            submissions = [
                os.path.join(target_submission_folder, item)
                for item in sorted(os.listdir(target_submission_folder))
                if os.path.isdir(os.path.join(target_submission_folder, item))
                and "submission" in item.lower()
            ]
            if not submissions:
                raise FileNotFoundError(
                    f"No SubmissionXXXX folders found under: {target_submission_folder}"
                )
    else:
        submissions = [
            os.path.join(forum_folder, item)
            for item in sorted(os.listdir(forum_folder))
            if os.path.isdir(os.path.join(forum_folder, item)) and "Submission" in item
        ]

    print(f"Found {len(submissions)} submissions to evaluate.")

    api_normalized = (api_provider or "azure").lower()
    if api_normalized == "azure":
        client = AzureOpenAIClient()
    elif api_normalized in {"openai", "openai-url", "openai_url", "pdf-url"}:
        client = OpenAIClient()
    else:
        raise ValueError(f"Unsupported evaluation API: {api_provider}")

    summary: Dict[str, object] = {
        "timestamp": datetime.now().isoformat(),
        "total_submissions": len(submissions),
        "successful": 0,
        "failed": 0,
        "decisions": {"REWRITE": 0, "OK": 0, "UNKNOWN": 0},
        "conflicts": {"YES": 0, "NO": 0, "UNKNOWN": 0},
        "results": [],
    }

    for idx, submission_path in enumerate(submissions, 1):
        submission_id = os.path.basename(submission_path)
        print(f"\n{'=' * 70}\nEvaluating {submission_id} ({idx}/{len(submissions)})\n{'=' * 70}")

        try:
            reviews = _collect_reviews(submission_path)
            if not reviews:
                raise RuntimeError("No review_*.txt files found.")

            meta_review_text, meta_review_path = _find_meta_review(
                submission_path,
                preferred_mode,
            )

            confidential_note = _load_confidential_note(submission_path)
            if confidential_note:
                print("  • Including submission_discussion.txt")
            else:
                print("  • No submission_discussion.txt")

            paper_title = _extract_paper_title(submission_path, meta_review_text, reviews)
            if paper_title:
                print(f"  • Title: {paper_title}")
            else:
                paper_title = submission_id

            raw_response = evaluate_meta_review(
                client=client,
                reviews=reviews,
                meta_review=meta_review_text,
                confidential_note=confidential_note,
                score_statement=score_statement,
                api_provider=api_normalized,
            )
            interpreted = interpret_evaluation_response(raw_response)
            decision = interpreted["decision"]
            rewrite_reason = interpreted["rewrite_reason"]
            conflict_flag = interpreted["conflict"]
            conflict_reason = interpreted["conflict_reason"]

            summary["decisions"].setdefault(decision, 0)
            summary["decisions"][decision] += 1
            summary["conflicts"].setdefault(conflict_flag, 0)
            summary["conflicts"][conflict_flag] += 1
            summary["successful"] = int(summary["successful"]) + 1

            result_file = os.path.join(output_folder, f"{submission_id}_meta_review_evaluation.txt")
            with open(result_file, "w", encoding="utf-8") as handle:
                handle.write(f"Submission: {submission_id}\n")
                handle.write(f"Title: {paper_title}\n")
                handle.write(f"Meta-review file: {meta_review_path}\n")
                handle.write(f"Assessment: {decision}\n")
                if rewrite_reason:
                    handle.write(f"Reason: {rewrite_reason}\n")
                handle.write(f"Conflict with reviews: {conflict_flag}\n")
                if conflict_reason:
                    handle.write(f"Conflict explanation: {conflict_reason}\n")
                handle.write("Raw Output:\n")
                handle.write(raw_response if raw_response.endswith("\n") else raw_response + "\n")

            summary["results"].append(
                {
                    "submission_id": submission_id,
                    "paper_title": paper_title,
                    "meta_review": meta_review_path,
                    "decision": decision,
                    "reason": rewrite_reason,
                    "conflict": conflict_flag,
                    "conflict_reason": conflict_reason,
                    "raw": raw_response,
                    "status": "success",
                    "output": result_file,
                }
            )

            if rewrite_reason:
                print(f"  • Rewrite check: {decision} – {rewrite_reason}")
            else:
                print(f"  • Rewrite check: {decision}")
            suffix = f" – {conflict_reason}" if conflict_reason else ""
            print(f"  • Conflict with reviews: {conflict_flag}{suffix}")
            print(f"  • Result saved to {result_file}")

        except Exception as exc:  # pylint: disable=broad-except
            summary["failed"] = int(summary["failed"]) + 1
            summary["results"].append(
                {
                    "submission_id": submission_id,
                    "status": "failed",
                    "error": str(exc),
                }
            )
            print(f"  ✗ Failed: {exc}")

    summary_path = os.path.join(output_folder, "batch_evaluation_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("ICLR 2026 Meta-Review Evaluation Summary\n")
        handle.write("=" * 60 + "\n")
        handle.write(f"Timestamp: {summary['timestamp']}\n")
        handle.write(f"Total submissions: {summary['total_submissions']}\n")
        handle.write(f"Successful: {summary['successful']}\n")
        handle.write(f"Failed: {summary['failed']}\n")
        handle.write("\nAssessment breakdown:\n")
        for decision, count in summary["decisions"].items():
            handle.write(f"  {decision}: {count}\n")
        handle.write("\nConflict breakdown:\n")
        for flag, count in summary["conflicts"].items():
            handle.write(f"  {flag}: {count}\n")
        handle.write("\nDetailed results:\n")
        handle.write("-" * 40 + "\n")
        for item in summary["results"]:
            handle.write(f"Submission: {item['submission_id']}\n")
            if item.get("paper_title"):
                handle.write(f"Title: {item['paper_title']}\n")
            handle.write(f"Status: {item['status']}\n")
            if item["status"] == "success":
                handle.write(f"Meta-review: {item['meta_review']}\n")
                handle.write(f"Assessment: {item['decision']}\n")
                if item.get("reason"):
                    handle.write(f"Reason: {item['reason']}\n")
                handle.write(f"Conflict: {item.get('conflict', 'UNKNOWN')}\n")
                if item.get("conflict_reason"):
                    handle.write(f"Conflict explanation: {item['conflict_reason']}\n")
                handle.write(f"Output: {item['output']}\n")
            else:
                handle.write(f"Error: {item['error']}\n")
            handle.write("-" * 20 + "\n")

    csv_path = os.path.join(output_folder, "batch_evaluation_summary.csv")
    with open(csv_path, "w", encoding="utf-8") as handle:
        handle.write(
            "Submission_ID,Title,Status,Assessment,Conflict,MetaReview_File,Output_File\n"
        )
        for item in summary["results"]:
            if item["status"] == "success":
                conflict_flag = item.get("conflict", "UNKNOWN")
                row = [
                    item["submission_id"],
                    (item.get("paper_title") or "").replace(",", " "),
                    "success",
                    item["decision"],
                    conflict_flag,
                    item["meta_review"],
                    item["output"],
                ]
                handle.write(",".join(row) + "\n")
            else:
                error_clean = item["error"].replace(",", ";").replace("\n", " ")
                row = [
                    item["submission_id"],
                    "",
                    "failed",
                    "UNKNOWN",
                    "",
                    error_clean,
                ]
                handle.write(",".join(row) + "\n")

    print(f"\nSummary written to {summary_path}")
    print(f"CSV written to {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch meta-review evaluation")
    parser.add_argument("forum_folder", help="Path to forum folder containing SubmissionXXXX directories")
    parser.add_argument(
        "--submission-folder",
        help="Optional path to a single SubmissionXXXX directory to evaluate",
    )
    parser.add_argument(
        "--mode",
        help="Meta-review generation mode name to match in filenames (e.g., balanced)",
    )
    parser.add_argument(
        "--output-folder",
        default="outputs/meta_review_evaluations",
        help=(
            "Directory to store meta-review evaluation outputs "
            "(default: outputs/meta_review_evaluations)"
        ),
    )
    parser.add_argument(
    "--api",
    choices=["azure", "openai"],
    default="azure",
    help="LLM provider to use for evaluation (default: azure)",
    )
    parser.add_argument(
        "--score-statement",
        help="Optional textual description of the reviewer scoring scale to include in prompts",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.forum_folder):
        raise FileNotFoundError(f"Forum folder not found: {args.forum_folder}")

    target_submission: Optional[str] = None
    if args.submission_folder:
        target_submission = args.submission_folder
        if not os.path.isabs(target_submission):
            target_submission = os.path.join(args.forum_folder, target_submission)
        if not os.path.isdir(target_submission):
            raise FileNotFoundError(f"Submission folder not found: {target_submission}")

    batch_evaluate_meta_reviews(
        forum_folder=args.forum_folder,
        target_submission_folder=target_submission,
        output_folder=args.output_folder,
        preferred_mode=args.mode,
        score_statement=args.score_statement or "",
        api_provider=args.api,
    )


if __name__ == "__main__":
    main()
