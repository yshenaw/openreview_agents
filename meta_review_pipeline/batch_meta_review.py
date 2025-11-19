#!/usr/bin/env python3
"""Batch pipeline to export OpenReview assignments and generate/evaluate meta-reviews."""

import argparse
import os
import shutil
from datetime import datetime
from typing import Dict, List, Optional

from crawler_pipeline.export_assigned_submissions import (
    build_client,
    collect_assignments,
    ensure_dir,
    export_submission_bundle,
)
from .generation.generate_meta_review import (
    MetaReviewGenerator,
    extract_recommendation,
    save_meta_review,
)
from .utils.export_process import extract_submission_data
from .evaluate.batch_evaluate_meta_review import batch_evaluate_meta_reviews


class CommaSeparatedValues(argparse.Action):
    """Custom argparse action to split comma-separated values with optional spaces."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: object,
        option_string: Optional[str] = None,
    ) -> None:
        current = getattr(namespace, self.dest, None) or []
        tokens: List[str]
        if isinstance(values, list):
            tokens = values
        else:
            tokens = [str(values)]

        for token in tokens:
            if not token:
                continue
            parts = [item.strip() for item in str(token).split(",")]
            current.extend([item for item in parts if item])

        setattr(namespace, self.dest, current)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download assigned submissions from OpenReview and auto-generate "
            "meta-reviews using the Azure/OpenAI workflow."
        )
    )
    parser.add_argument("venue_id", help="Parent venue id, e.g. ICLR.cc/2026/Conference", default = "ICLR.cc/2026/Conference")
    parser.add_argument(
        "--baseurl",
        default="https://api2.openreview.net",
        help="OpenReview API base URL (default: https://api2.openreview.net)",
    )
    parser.add_argument(
        "--username",
        default=os.environ.get("OPENREVIEW_USERNAME"),
        help="OpenReview username/email (defaults to OPENREVIEW_USERNAME env)",
    )
    parser.add_argument(
        "--password",
        default=os.environ.get("OPENREVIEW_PASSWORD"),
        help="OpenReview password (defaults to OPENREVIEW_PASSWORD env)",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("OPENREVIEW_TOKEN"),
        help="OpenReview personal token; overrides username/password when provided",
    )
    parser.add_argument(
        "--role",
        choices=["author", "reviewer", "ac", "sac", "audience"],
        default="ac",
        help="Assignment scope for exports (author, reviewer, AC, SAC, or audience)",
    )
    parser.add_argument(
        "--audience-paper-type",
        choices=["oral", "spotlight", "poster", "rejected"],
        default=None,
        help=(
            "Audience-only: Acceptance category filter. "
            "Defaults to poster when not provided and no forum IDs are given."
        ),
    )
    parser.add_argument(
        "--forum-id",
        dest="forum_ids",
        action=CommaSeparatedValues,
        nargs="+",
        help=(
            "Audience-only: Explicit forum ID to include. "
            "May be provided multiple times or as a comma-separated list."
        ),
    )
    parser.add_argument(
        "--api",
        choices=["azure", "openai"],
        default="azure",
        help="LLM provider to use for both generation and evaluation (default: azure)",
    )
    parser.add_argument(
        "--download-dir",
        default=os.path.join("outputs", "openreview_exports"),
        help="Destination folder for exported submissions",
    )
    parser.add_argument(
        "--meta-output-dir",
        default=os.path.join("outputs", "generated_meta_reviews"),
        help="Where to store generated meta-review files",
    )
    parser.add_argument(
        "--run-tag",
        help=(
            "Optional label for this run; also used as the subdirectory name "
            "inside download and output folders"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Process at most N assigned submissions (useful for smoke tests)",
    )
    parser.add_argument(
        "--skip-existing-export",
        action="store_true",
        help="Reuse previously exported folders when present",
    )
    parser.add_argument(
        "--task",
        choices=["generate", "evaluate", "both"],
        default="generate",
        help="Pipeline action: generate meta-reviews, evaluate existing ones, or both",
    )
    parser.add_argument(
        "--submission-folder",
        help=(
            "Optional path to an existing SubmissionXXXX directory or a directory containing them. "
            "When provided, the pipeline reuses those exports instead of downloading from OpenReview."
        ),
    )
    parser.add_argument(
        "--meta-review-folder",
        help="Override meta-review folder when evaluating without generation",
    )
    parser.add_argument(
        "--no-rebuttal",
        action="store_true",
        help=(
            "Add guidance to prompts that the decision is before rebuttal and must rely solely on the paper."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generation_mode = "balanced"
    raw_forum_ids: List[str] = []
    if args.forum_ids:
        for value in args.forum_ids:
            if not value:
                continue
            parts = [item.strip() for item in value.split(",")]
            raw_forum_ids.extend([item for item in parts if item])
    args.forum_ids = raw_forum_ids or None

    if args.role == "audience":
        if not args.forum_ids and not args.audience_paper_type:
            args.audience_paper_type = "poster"
        if args.limit is None and not args.forum_ids:
            args.limit = 5

    submission_override: Optional[str] = None
    submission_override_single: Optional[str] = None
    submission_override_root: Optional[str] = None
    submission_dirs: List[str] = []

    if args.submission_folder:
        submission_override = os.path.abspath(args.submission_folder)
        if not os.path.isdir(submission_override):
            raise FileNotFoundError(f"Submission folder not found: {submission_override}")

        base_name = os.path.basename(submission_override.rstrip(os.sep)).lower()
        if "submission" in base_name:
            submission_override_single = submission_override
            submission_override_root = os.path.dirname(submission_override) or os.path.abspath(".")
            submission_dirs = [submission_override_single]
        else:
            submission_override_root = submission_override
            submission_dirs = [
                os.path.join(submission_override, item)
                for item in sorted(os.listdir(submission_override))
                if os.path.isdir(os.path.join(submission_override, item))
                and "submission" in item.lower()
            ]
            if not submission_dirs:
                raise FileNotFoundError(
                    f"No SubmissionXXXX folders found under: {submission_override}"
                )

        assigned: List[Dict] = []
        total_assigned = len(submission_dirs)
        target_count = total_assigned
        client = None
    else:
        client = build_client(
            baseurl=args.baseurl,
            username=args.username,
            password=args.password,
            token=args.token,
            require_login=args.role != "audience",
        )

        assigned, total_assigned = collect_assignments(
            client=client,
            venue_id=args.venue_id,
            role=args.role,
            audience_paper_type=args.audience_paper_type,
            forum_ids=args.forum_ids,
            limit=args.limit,
        )
        if not assigned:
            print("No submissions are currently assigned to this profile.")
            return

        target_count = len(assigned)

    tasks_generate = args.task in {"generate", "both"}
    tasks_evaluate = args.task in {"evaluate", "both"}

    if submission_override:
        download_root = submission_override_root or submission_override
        run_tag = args.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        run_tag = args.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
        download_root = ensure_dir(os.path.join(args.download_dir, run_tag))

    meta_output_dir: Optional[str]
    meta_source_dir: Optional[str]

    if tasks_generate:
        meta_output_dir = args.meta_review_folder or os.path.join(
            args.meta_output_dir, run_tag
        )
        ensure_dir(meta_output_dir)
        if args.meta_review_folder:
            meta_source_dir = args.meta_review_folder
        elif submission_override:
            meta_source_dir = submission_override
        else:
            meta_source_dir = download_root
    else:
        meta_output_dir = None
        meta_source_dir = args.meta_review_folder

    generator = MetaReviewGenerator(api=args.api) if tasks_generate else None
    score_statement = (
        "The score is indicated by the overall recommendation on a scale from 1 to 5, "
        "with 1 being the lowest and 5 being the highest. 2 is the borderline reject and 3 is the borderline accept."
    )

    stats: Dict[str, object] = {
        "timestamp": datetime.now().isoformat(),
        "venue_id": args.venue_id,
        "role": args.role,
        "total_assigned": total_assigned,
        "processed": 0,
        "generated": 0,
        "failed": 0,
        "recommendations": {"Oral": 0, "Spotlight": 0, "Poster": 0, "Reject": 0, "Unknown": 0},
        "results": [],
    }

    if submission_override:
        print(
            f"Processing {target_count} submission folder(s) "
            f"provided via --submission-folder as {args.role.upper()}."
        )
        print(f"Submission root: {submission_override}")
        if tasks_generate and meta_output_dir:
            print(f"Generated meta-reviews will be saved under: {meta_output_dir}")
            print("Copies will also be placed inside each submission folder.\n")
        elif meta_source_dir:
            print(f"Meta-review source: {meta_source_dir}\n")
        else:
            print("Meta-review source: will search existing files inside each submission folder.\n")
    else:
        print(
            f"Processing {target_count} assigned submissions "
            f"(out of {total_assigned}) as {args.role.upper()}."
        )
        print(f"Exports will be saved under: {download_root}")
        if tasks_generate and meta_output_dir:
            print(f"Generated meta-reviews will be saved under: {meta_output_dir}")
            print("Copies will also be placed inside each submission folder for evaluation.\n")
        elif meta_source_dir:
            print(f"Using meta-reviews from: {meta_source_dir}\n")
        else:
            print("Meta-review source: will search existing files inside each submission folder.\n")

    export_records: List[Dict[str, object]] = []

    if not submission_override:
        for entry in assigned:
            paper_number = entry["number"]
            submission_id = f"Submission{paper_number}"
            title = entry.get("title", "(untitled)")

            print("=" * 80)
            print(f"Paper #{paper_number}: {title}")
            print("=" * 80)

            record: Dict[str, object] = {
                "submission_id": submission_id,
                "title": title,
                "paper_number": paper_number,
                "export_dir": None,
                "export_reused": False,
                "export_status": "pending",
                "error": None,
            }

            try:
                export_dir, reused = export_submission_bundle(
                    client=client,
                    entry=entry,
                    download_root=download_root,
                    skip_existing_export=args.skip_existing_export,
                )
                record["export_dir"] = export_dir
                record["export_reused"] = reused
                record["export_status"] = "success"
                if reused:
                    print(f"  • Reusing existing export at {export_dir}")
                else:
                    print(f"  • Exported to {export_dir}")
            except Exception as exc:  # pylint: disable=broad-except
                record["export_status"] = "failed"
                record["error"] = str(exc)
                print(f"  ✗ Export failed: {exc}\n")

            export_records.append(record)
    else:
        for submission_dir in submission_dirs:
            submission_id = os.path.basename(submission_dir.rstrip(os.sep))
            print("=" * 80)
            print(f"Submission folder: {submission_id}")
            print("=" * 80)

            record = {
                "submission_id": submission_id,
                "title": submission_id,
                "paper_number": None,
                "export_dir": submission_dir,
                "export_reused": True,
                "export_status": "success",
                "error": None,
            }
            export_records.append(record)

    if tasks_generate and generator:
        if export_records:
            print("\nStarting meta-review generation phase...\n")

        for record in export_records:
            submission_id = str(record["submission_id"])
            title = str(record["title"])
            export_dir = record.get("export_dir")

            if record.get("export_status") != "success" or not export_dir:
                if tasks_generate:
                    stats["results"].append(
                        {
                            "submission_id": submission_id,
                            "title": title,
                            "status": "failed",
                            "error": record.get("error", "Export did not complete."),
                        }
                    )
                    stats["failed"] = int(stats["failed"]) + 1
                stats["processed"] = int(stats["processed"]) + 1
                continue

            print("-" * 80)
            print(f"Generating meta-review for {submission_id}: {title}")

            try:
                submission_data = extract_submission_data(
                    str(export_dir),
                    convert_to_images=(generator.provider == "azure"),
                )
                if not submission_data.get("reviews"):
                    raise RuntimeError("No review_*.txt files found after export")

                submission_discussion = submission_data.get("submission_discussion")
                meta_text = generator.generate_meta_review(
                    paper_images=submission_data.get("paper_images", []),
                    reviews=submission_data["reviews"],
                    score_statement=score_statement,
                    confidential_note=submission_discussion,
                    forum_id=submission_data.get("forum_id"),
                    paper_pdf=submission_data.get("paper_pdf"),
                    no_rebuttal=args.no_rebuttal,
                )

                recommendation = extract_recommendation(meta_text)
                stats["recommendations"].setdefault(recommendation, 0)
                stats["recommendations"][recommendation] += 1

                if not meta_output_dir:
                    raise RuntimeError("Meta-review output directory was not initialized.")

                output_path = os.path.join(
                    meta_output_dir, f"{submission_id}_generated_meta_review.txt"
                )
                generated_path = save_meta_review(
                    submission_id=submission_data["submission_id"],
                    meta_review_content=meta_text,
                    submission_data=submission_data,
                    mode=generation_mode,
                    output_path=output_path,
                )

                export_copy_path = os.path.join(
                    str(export_dir), os.path.basename(generated_path)
                )
                if os.path.abspath(generated_path) != os.path.abspath(export_copy_path):
                    shutil.copy2(generated_path, export_copy_path)

                stats["results"].append(
                    {
                        "submission_id": submission_id,
                        "title": title,
                        "status": "success",
                        "recommendation": recommendation,
                        "meta_review_file": generated_path,
                        "submission_meta_review_file": export_copy_path,
                        "export_dir": export_dir,
                    }
                )
                stats["generated"] = int(stats["generated"]) + 1
                print(f"  • Meta-review saved to {generated_path}")
                if os.path.abspath(generated_path) != os.path.abspath(export_copy_path):
                    print(f"  • Meta-review copied to {export_copy_path}")
                print(f"  • Recommendation: {recommendation}\n")

            except Exception as exc:  # pylint: disable=broad-except
                stats["results"].append(
                    {
                        "submission_id": submission_id,
                        "title": title,
                        "status": "failed",
                        "error": str(exc),
                    }
                )
                stats["failed"] = int(stats["failed"]) + 1
                print(f"  ✗ Meta-review generation failed: {exc}\n")

            finally:
                stats["processed"] = int(stats["processed"]) + 1
    else:
        for record in export_records:
            if record.get("export_status") != "success":
                stats["failed"] = int(stats["failed"]) + 1
            stats["processed"] = int(stats["processed"]) + 1

    evaluation_output_dir: Optional[str] = None
    evaluation_summary_path: Optional[str] = None
    if tasks_evaluate:
        evaluation_parent = meta_source_dir or download_root
        evaluation_output_dir = os.path.join(evaluation_parent, "meta_review_evaluations")
        if submission_override:
            if submission_override_single:
                evaluation_forum_folder = submission_override_root or os.path.dirname(submission_override_single) or submission_override_single
                evaluation_target_folder = submission_override_single
            else:
                evaluation_forum_folder = submission_override
                evaluation_target_folder = None
        else:
            evaluation_forum_folder = download_root
            evaluation_target_folder = None

        print("=" * 80)
        print("Running meta-review evaluations...")
        try:
            batch_evaluate_meta_reviews(
                forum_folder=evaluation_forum_folder,
                target_submission_folder=evaluation_target_folder,
                output_folder=evaluation_output_dir,
                preferred_mode=None,
                score_statement=score_statement,
                api_provider=args.api,
            )
            evaluation_summary_path = os.path.join(
                evaluation_output_dir, "batch_evaluation_summary.txt"
            )
        except Exception as exc:  # pylint: disable=broad-except
            evaluation_output_dir = None
            evaluation_summary_path = None
            print(f"Meta-review evaluation failed: {exc}")
        print("=" * 80)

    if tasks_generate and meta_output_dir:
        summary_path = os.path.join(meta_output_dir, "batch_meta_review_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as handle:
            handle.write("ICLR 2026 Batch Meta-Review Summary\n")
            handle.write("=" * 60 + "\n")
            handle.write(f"Timestamp: {stats['timestamp']}\n")
            handle.write(f"Venue: {stats['venue_id']}\n")
            handle.write(f"Role: {stats['role']}\n")
            handle.write(f"Assigned (total): {stats['total_assigned']}\n")
            handle.write(f"Processed: {stats['processed']}\n")
            handle.write(f"Generated: {stats['generated']}\n")
            handle.write(f"Failed: {stats['failed']}\n")
            handle.write("\nRecommendation breakdown:\n")
            for rec, count in stats["recommendations"].items():
                if count:
                    handle.write(f"  {rec}: {count}\n")
            handle.write("\nResults:\n")
            handle.write("-" * 40 + "\n")
            for item in stats["results"]:
                handle.write(f"Submission: {item['submission_id']}\n")
                handle.write(f"Title: {item['title']}\n")
                handle.write(f"Status: {item['status']}\n")
                if item["status"] == "success":
                    handle.write(f"Recommendation: {item['recommendation']}\n")
                    handle.write(f"Meta-review file: {item['meta_review_file']}\n")
                    if item.get("submission_meta_review_file"):
                        handle.write(
                            f"Submission folder meta-review: {item['submission_meta_review_file']}\n"
                        )
                    handle.write(f"Export dir: {item['export_dir']}\n")
                else:
                    handle.write(f"Error: {item['error']}\n")
                handle.write("-" * 20 + "\n")

            if evaluation_output_dir:
                handle.write("\nMeta-Review Evaluation Outputs:\n")
                handle.write(f"  Output directory: {evaluation_output_dir}\n")
                if evaluation_summary_path and os.path.exists(evaluation_summary_path):
                    handle.write(f"  Summary file: {evaluation_summary_path}\n")

        print("=" * 80)
        print("Batch meta-review generation complete.")
        print(f"Summary written to {summary_path}")
    else:
        print("=" * 80)
        print("Export step complete.")

    if evaluation_summary_path and os.path.exists(evaluation_summary_path):
        print(f"Meta-review evaluation summary: {evaluation_summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
