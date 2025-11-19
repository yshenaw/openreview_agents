#!/usr/bin/env python3
"""Utilities and CLI for exporting assigned OpenReview submissions."""

from __future__ import annotations

import argparse
import getpass
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from openreview import api as openreview_api

from .assignment_fetch import (
    get_papers,
    get_papers_author,
    get_papers_audience,
    get_papers_reviewer,
    get_papers_sac,
)
from .forum_exporter import export_forum_threads_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download assigned submissions from OpenReview without generating meta-reviews."
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
        help="Assignment role to fetch (Author, Reviewer, AC, SAC, or Audience)",
    )
    parser.add_argument(
        "--audience-paper-type",
        choices=["oral", "spotlight", "poster", "rejected"],
        default=None,
        help=(
            "Audience-only: Acceptance category filter. "
            "Defaults to poster when not provided and no forum IDs are specified."
        ),
    )
    parser.add_argument(
        "--forum-id",
        dest="forum_ids",
        action="append",
        help=(
            "Audience-only: Explicit forum ID to include. "
            "May be provided multiple times or as a comma-separated list."
        ),
    )
    parser.add_argument(
        "--download-dir",
        default=os.path.join("outputs", "openreview_exports"),
        help="Destination folder for exported submissions",
    )
    parser.add_argument(
        "--run-tag",
        help=(
            "Optional label for this run; also used as the subdirectory name inside the download folder"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Export at most N assigned submissions (useful for smoke tests)",
    )
    parser.add_argument(
        "--skip-existing-export",
        action="store_true",
        help="Reuse previously exported folders when present",
    )
    return parser.parse_args()


def build_client(
    baseurl: str,
    username: Optional[str] = None,
    password: Optional[str] = None,
    token: Optional[str] = None,
    require_login: bool = True,
) -> openreview_api.OpenReviewClient:
    """Construct an OpenReview client, prompting for credentials when needed."""

    if not require_login:
        return openreview_api.OpenReviewClient(baseurl=baseurl)

    if token:
        return openreview_api.OpenReviewClient(baseurl=baseurl, token=token)

    user = username or input("OpenReview username/email: ").strip()
    secret = password or getpass.getpass("OpenReview password: ")

    if not user or not secret:
        raise ValueError("Valid OpenReview credentials are required")

    return openreview_api.OpenReviewClient(
        baseurl=baseurl,
        username=user,
        password=secret,
    )


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def collect_assignments(
    client: openreview_api.OpenReviewClient,
    venue_id: str,
    role: str,
    *,
    audience_paper_type: Optional[str] = None,
    forum_ids: Optional[List[str]] = None,
    limit: Optional[int] = None,
) -> Tuple[List[Dict[str, object]], int]:
    """Fetch and sort assignments for the requested role."""

    if role == "sac":
        assigned = get_papers_sac(client, venue_id)
    elif role == "ac":
        assigned = get_papers(client, venue_id)
    elif role == "reviewer":
        assigned = get_papers_reviewer(client, venue_id)
    elif role == "author":
        assigned = get_papers_author(client, venue_id)
    elif role == "audience":
        matches = get_papers_audience(
            client,
            venue_id,
            audience_paper_type,
            forum_ids=forum_ids,
        )
        total_assigned = len(matches)
        if limit is not None:
            matches = matches[:limit]
        return matches, total_assigned
    else:
        raise ValueError(f"Unsupported role: {role}")

    total_assigned = len(assigned)
    assigned.sort(key=lambda item: item["number"])

    if limit is not None:
        assigned = assigned[:limit]

    return assigned, total_assigned


def export_submission_bundle(
    client: openreview_api.OpenReviewClient,
    entry: Dict[str, object],
    download_root: str,
    skip_existing_export: bool = False,
    *,
    include_paper_pdf: bool = True,
    prefer_latest_pdf: bool = True,
) -> Tuple[str, bool]:
    """Export a single submission bundle; return (export_dir, reused_flag)."""

    submission_id = f"Submission{entry['number']}"
    target_dir = os.path.join(download_root, submission_id)

    if skip_existing_export and os.path.isdir(target_dir):
        return target_dir, True

    export_result = export_forum_threads_text(
        forum_id=entry["forum_id"],
        client=client,
        outdir=download_root,
        include_paper_pdf=include_paper_pdf,
        prefer_latest_pdf=prefer_latest_pdf,
    )
    return export_result["paper_dir"], False


def export_assigned_submissions(
    client: openreview_api.OpenReviewClient,
    assigned: List[Dict[str, object]],
    download_root: str,
    skip_existing_export: bool = False,
) -> Dict[str, object]:
    """Export all assigned submissions and return basic run statistics."""

    ensure_dir(download_root)

    stats: Dict[str, object] = {
        "timestamp": datetime.now().isoformat(),
        "download_root": download_root,
        "processed": 0,
        "exported": 0,
        "reused": 0,
        "failed": 0,
        "results": [],
    }

    for entry in assigned:
        submission_id = f"Submission{entry['number']}"
        try:
            export_dir, reused = export_submission_bundle(
                client=client,
                entry=entry,
                download_root=download_root,
                skip_existing_export=skip_existing_export,
            )
            stats["results"].append(
                {
                    "submission_id": submission_id,
                    "title": entry.get("title", ""),
                    "export_dir": export_dir,
                    "reused": reused,
                    "status": "success",
                }
            )
            stats["exported" if not reused else "reused"] = int(
                stats["exported" if not reused else "reused"]
            ) + 1
        except Exception as exc:  # pylint: disable=broad-except
            stats["results"].append(
                {
                    "submission_id": submission_id,
                    "title": entry.get("title", ""),
                    "status": "failed",
                    "error": str(exc),
                }
            )
            stats["failed"] = int(stats["failed"]) + 1

        stats["processed"] = int(stats["processed"]) + 1

    return stats


def main() -> None:
    args = parse_args()
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

    run_tag = args.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    download_root = ensure_dir(os.path.join(args.download_dir, run_tag))

    print(
        f"Exporting {len(assigned)} submissions (out of {total_assigned}) as {args.role.upper()} to {download_root}."
    )

    stats = export_assigned_submissions(
        client=client,
        assigned=assigned,
        download_root=download_root,
        skip_existing_export=args.skip_existing_export,
    )

    print("\nExport summary:")
    print(f"  Processed: {stats['processed']}")
    print(f"  Exported: {stats['exported']}")
    print(f"  Reused: {stats['reused']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Output directory: {download_root}")


if __name__ == "__main__":
    main()
