"""Utilities for exporting OpenReview forums to local text bundles."""

import os
import re
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional


def export_forum_threads_text(
    forum_id: str,
    client,
    outdir: str = "forum_export",
    include_paper_pdf: bool = True,
    prefer_latest_pdf: bool = True,
    pdf_filename: Optional[str] = None,
    review_name_override: Optional[str] = None,
    include_meta_review: bool = True,
    meta_review_mode: str = "per_paper_file",
) -> Dict[str, Any]:
    """Export a single paper's review threads in plain text."""

    def _sanitize_filename(name: str, max_len: int = 80) -> str:
        name = re.sub(r"[^\w\-\s]", "", name).strip()
        return (name[:max_len]).replace(" ", "_")

    def _fmt_ts(ms: int) -> str:
        return datetime.utcfromtimestamp(ms / 1000).isoformat() + "Z"

    def _content_value(note, key: str):
        c = (note.content or {}).get(key, {}) or {}
        return c.get("value")

    def _content_text(note, keys: List[str]) -> str:
        for k in keys:
            val = _content_value(note, k)
            if val is None:
                continue
            if isinstance(val, list):
                text = "\n".join(str(x) for x in val if str(x).strip())
                if text.strip():
                    return text
            else:
                s = str(val).strip()
                if s:
                    return s
        return ""

    def _inv_list(note):
        inv = getattr(note, "invitation", None)
        invs = getattr(note, "invitations", None)
        if invs and isinstance(invs, list):
            return invs
        return [inv] if inv else []

    def _has_inv_tail(note, tail: str) -> bool:
        return any(("/-/" + tail) in i or i.endswith(tail) for i in _inv_list(note))

    def _is_official_comment(note) -> bool:
        return _has_inv_tail(note, "Official_Comment")

    def _is_rebuttal(note) -> bool:
        keywords = (
            "rebuttal",
            "official_response",
            "author_response",
            "author_official_comment",
            "reply_rebuttal",
            "author_discussion",
            "author_reply",
            "author_ac_confidential_comments",
        )
        for inv in _inv_list(note):
            if not inv:
                continue
            tail = inv.split("/-/")[-1]
            tail_lower = tail.lower()
            if any(key in tail_lower for key in keywords):
                return True
            if "confidential_comment" in tail_lower and "author" in tail_lower:
                return True
        return False

    def _is_reviewer_official_comment(note, reviewers_group_id: str, venue_id: str, submission) -> bool:
        if not _is_official_comment(note):
            return False
        sigs = getattr(note, "signatures", []) or []
        return any(
            (reviewers_group_id in s)
            or (f"{venue_id}/Submission{submission.number}/Reviewer_" in s)
            or ("Anonymous" in s)
            or ("AnonReviewer" in s)
            for s in sigs
        )

    def _is_official_review(note, review_name: str, venue_id: str, submission) -> bool:
        if _has_inv_tail(note, review_name or "Official_Review"):
            if getattr(note, "replyto", None) == submission.id:
                return True

        if getattr(note, "replyto", None) != submission.id:
            return False

        sigs = getattr(note, "signatures", []) or []
        is_reviewer_sig = any(
            (f"{venue_id}/Submission{submission.number}/Reviewers" in s)
            or (f"{venue_id}/Submission{submission.number}/Reviewer_" in s)
            or ("Anonymous" in s)
            or ("AnonReviewer" in s)
            for s in sigs
        )
        if not is_reviewer_sig:
            return False

        typical_review_keys = {
            "summary",
            "summary_of_the_paper",
            "main_review",
            "strengths",
            "weaknesses",
            "questions",
            "rating",
            "confidence",
            "soundness",
            "presentation",
            "contribution",
        }
        has_review_fields = any(k in (note.content or {}) for k in typical_review_keys)
        return has_review_fields

    def _is_meta_review(note) -> bool:
        return _has_inv_tail(note, "Meta_Review")

    def _is_decision(note) -> bool:
        tails = (
            "Decision",
            "Paper_Decision",
            "Submission_Decision",
            "Meta_Review_Decision",
        )
        return any(_has_inv_tail(note, tail) for tail in tails)

    def _render_meta_review(out, mr):
        out.write("META-REVIEW\n")
        out.write("-" * 80 + "\n")
        out.write(f"Signatures: {', '.join(getattr(mr, 'signatures', []) or [])}\n")
        out.write(f"Posted: { _fmt_ts(getattr(mr, 'cdate', 0)) }\n\n")

        blocks = [
            ("Metareview", ["metareview"]),
            ("Additional comments on reviewer discussion", ["additional_comments_on_reviewer_discussion"]),
        ]
        printed = set()
        for label, keys in blocks:
            txt = _content_text(mr, keys)
            if txt:
                out.write(f"{label}:\n{txt}\n\n")
                printed.update(k for k in keys if (mr.content or {}).get(k))

        skip = set(printed)
        for k in list((mr.content or {}).keys()):
            if k in skip:
                continue
            val = _content_value(mr, k)
            if val is None:
                continue
            if isinstance(val, list):
                text = "\n".join(str(x) for x in val if str(x).strip())
            else:
                text = str(val).strip()
            if text:
                out.write(f"{k.replace('_', ' ').capitalize()}:\n{text}\n\n")

    def _render_decision_note(out, decision_note):
        out.write("PAPER DECISION\n")
        out.write("-" * 80 + "\n")
        out.write(f"Signatures: {', '.join(getattr(decision_note, 'signatures', []) or [])}\n")
        out.write(f"Posted: { _fmt_ts(getattr(decision_note, 'cdate', 0)) }\n\n")

        blocks = [
            ("Decision", ["decision", "recommendation", "final_decision"]),
            ("Meta-review", ["metareview", "meta_review", "meta-review", "commentary", "comment"]),
            ("Rationale", ["justification", "explanation", "supporting_text"]),
            ("Confidence", ["confidence", "decision_confidence"]),
        ]

        printed = set()
        for label, keys in blocks:
            txt = _content_text(decision_note, keys)
            if txt:
                out.write(f"{label}:\n{txt}\n\n")
                printed.update(k for k in keys if (decision_note.content or {}).get(k))

        skip = set(printed)
        for k in list((decision_note.content or {}).keys()):
            if k in skip:
                continue
            val = _content_value(decision_note, k)
            if val is None:
                continue
            if isinstance(val, list):
                text = "\n".join(str(x) for x in val if str(x).strip())
            else:
                text = str(val).strip()
            if text:
                out.write(f"{k.replace('_', ' ').capitalize()}:\n{text}\n\n")

    def _build_default_pdf_name(note, forum_identifier: str) -> str:
        title = _content_value(note, "title") or ""
        safe_title = _sanitize_filename(title) or "paper"
        safe_forum = _sanitize_filename(forum_identifier, max_len=40) or "forum"
        return f"{safe_forum}_{safe_title}.pdf"

    def _try_download_pdf(
        pdf_note,
        dest_dir: str,
        filename: Optional[str],
        forum_identifier: str,
    ) -> Optional[str]:
        has_pdf = _content_value(pdf_note, "pdf")
        if not has_pdf:
            return None
        try:
            pdf_bytes = client.get_attachment(field_name="pdf", id=pdf_note.id)
            os.makedirs(dest_dir, exist_ok=True)
            fname = filename or _build_default_pdf_name(pdf_note, forum_identifier)
            out_path = os.path.abspath(os.path.join(dest_dir, fname))
            with open(out_path, "wb") as f:
                f.write(pdf_bytes)
            return out_path
        except Exception as ex:  # pragma: no cover - relies on OpenReview
            print(f"[WARN] Could not download PDF from note {pdf_note.id}: {ex}")
            return None

    def _render_review_sections(out, review_note, ordered_schema_keys: Optional[List[str]] = None):
        def _get_rating_text(rating_value: str) -> str:
            if not rating_value or rating_value == "N/A":
                return "N/A"
            return str(rating_value).strip()

        def _get_confidence_text(confidence_value: str) -> str:
            if not confidence_value or confidence_value == "N/A":
                return "N/A"
            return str(confidence_value).strip()

        rating = _content_text(review_note, ["rating", "recommendation"])
        confidence = _content_text(review_note, ["confidence"])
        if rating or confidence:
            rating_with_text = _get_rating_text(rating) if rating else "N/A"
            confidence_with_text = _get_confidence_text(confidence) if confidence else "N/A"
            out.write(f"Rating: {rating_with_text} | Confidence: {confidence_with_text}\n\n")

        preferred = [
            ("TITLE", ["title"]),
            ("SUMMARY", ["summary", "summary_of_the_paper", "main_review"]),
            ("STRENGTHS", ["strengths", "what_are_the_strengths"]),
            ("WEAKNESSES", ["weaknesses", "what_are_the_weaknesses", "limitations"]),
            ("QUESTIONS", ["questions", "questions_to_authors"]),
        ]
        printed = set()

        for label, keys in preferred:
            txt = _content_text(review_note, keys)
            if txt:
                out.write(f"{label}:\n{txt}\n\n")
                printed.update(k for k in keys if (review_note.content or {}).get(k))

        facets = {
            "Soundness": _content_value(review_note, "soundness"),
            "Presentation": _content_value(review_note, "presentation"),
            "Contribution": _content_value(review_note, "contribution"),
        }
        if any(v is not None for v in facets.values()):
            line = " | ".join(f"{k}: {v}" for k, v in facets.items() if v is not None)
            out.write(f"{line}\n\n")

        skip = {"rating", "confidence", "soundness", "presentation", "contribution", "flag_for_ethics_review", "code_of_conduct"}
        keys_to_scan = ordered_schema_keys or list((review_note.content or {}).keys())
        for k in keys_to_scan:
            if k in skip or k in printed:
                continue
            val = _content_value(review_note, k)
            if val is None:
                continue
            if isinstance(val, list):
                text = "\n".join(str(x) for x in val if str(x).strip())
            else:
                text = str(val).strip()
            if text:
                out.write(f"{k.replace('_', ' ').capitalize()}:\n{text}\n\n")
                printed.add(k)

    def _comment_role(note, authors_group: str, reviewers_group: str, venue_id: str, submission) -> str:
        sigs = getattr(note, "signatures", []) or []
        if any(authors_group in s for s in sigs):
            return "Authors"
        rev = [s for s in sigs if (reviewers_group in s) or (f"{venue_id}/Submission{submission.number}/Reviewer_" in s)]
        if rev:
            s = rev[0]
            if "/Reviewer_" in s:
                return s.split("/")[-1]
            return "Reviewers"
        if any("Area_Chair" in s for s in sigs):
            return "Area Chair"
        if any("Senior_Area_Chairs" in s for s in sigs):
            return "Senior Area Chair"
        if any("Program_Chairs" in s for s in sigs):
            return "Program Chairs"
        return sigs[0] if sigs else "User"

    def _comment_text(note) -> str:
        return _content_text(
            note,
            [
                "comment",
                "text",
                "rebuttal",
                "author_response",
                "official_response",
                "response",
                "reply",
                "message",
            ],
        ) or "(no text)"

    def _write_discussion_thread(out, node, children_map, authors_group, reviewers_group, venue_id, submission, depth=0, index_path=""):
        indent = "  " * depth
        role = _comment_role(node, authors_group, reviewers_group, venue_id, submission)
        when = _fmt_ts(getattr(node, "cdate", 0))
        header = f"{indent}[{index_path}] {role} — Posted: {when}"
        out.write(header + "\n\n")
        out.write(indent + _comment_text(node).replace("\n", "\n" + indent) + "\n\n")
        for i, child in enumerate(children_map.get(node.id, []), start=1):
            child_idx = f"{index_path}.{i}" if index_path else str(i)
            _write_discussion_thread(out, child, children_map, authors_group, reviewers_group, venue_id, submission, depth + 1, child_idx)

    submission = client.get_note(forum_id)
    paper_number = submission.number
    title = _content_value(submission, "title") or ""
    short_title = _sanitize_filename(title)

    venue_id = getattr(submission, "domain", None) or _content_value(submission, "venueid")
    if not venue_id and getattr(submission, "invitation", None):
        venue_id = submission.invitations.split("/-/")[0]
    if not venue_id:
        tmp_notes = client.get_all_notes(forum=submission.id)
        for n in tmp_notes:
            for i in _inv_list(n):
                if "/Submission" in i:
                    venue_id = i.split("/Submission")[0]
                    break
            if venue_id:
                break
    if not venue_id:
        raise RuntimeError("Could not infer venue_id for this forum.")

    review_name = review_name_override or "Official_Review"
    try:
        venue_group = client.get_group(venue_id)
        review_name = (venue_group.content or {}).get("review_name", {}).get("value", review_name)
    except Exception:  # pragma: no cover - depends on OpenReview
        pass

    authors_group = f"{venue_id}/Submission{submission.number}/Authors"
    reviewers_group = f"{venue_id}/Submission{submission.number}/Reviewers"

    paper_dir = os.path.join(outdir, f"Submission{paper_number}")
    os.makedirs(paper_dir, exist_ok=True)

    paper_pdf_path: Optional[str] = None
    meta_review_paths: List[str] = []

    if include_paper_pdf:
        try:
            forum_notes_for_pdf = client.get_all_notes(forum=submission.id)
            if prefer_latest_pdf:
                forum_notes_sorted = sorted(
                    forum_notes_for_pdf, key=lambda n: getattr(n, "cdate", 0), reverse=True
                )
                for n in forum_notes_sorted:
                    paper_pdf_path = _try_download_pdf(
                        n,
                        paper_dir,
                        pdf_filename,
                        forum_id,
                    )
                    if paper_pdf_path:
                        break
            if paper_pdf_path is None:
                paper_pdf_path = _try_download_pdf(
                    submission,
                    paper_dir,
                    pdf_filename,
                    forum_id,
                )
        except Exception as ex:  # pragma: no cover - depends on OpenReview
            print(f"[WARN] PDF retrieval failed, continuing without PDF: {ex}")

    all_forum_notes = client.get_all_notes(forum=submission.id)
    replies = [n for n in all_forum_notes if n.id != submission.id]

    reviews = [
        n for n in replies
        if _is_official_review(n, review_name, venue_id, submission)
    ]

    discussion_notes = [n for n in replies if _is_official_comment(n) or _is_rebuttal(n)]
    meta_review_notes = [n for n in replies if _is_meta_review(n) and getattr(n, "replyto", None) == submission.id]
    meta_review_notes.sort(key=lambda n: getattr(n, "cdate", 0))

    decision_notes = [n for n in replies if _is_decision(n) and getattr(n, "replyto", None) == submission.id]
    decision_notes.sort(key=lambda n: getattr(n, "cdate", 0))

    meta_entries = [(note, "meta") for note in meta_review_notes]
    meta_entries.extend((note, "decision") for note in decision_notes)
    meta_entries.sort(key=lambda pair: getattr(pair[0], "cdate", 0))

    children_map: Dict[str, List[Any]] = defaultdict(list)
    for oc in discussion_notes:
        parent_id = getattr(oc, "replyto", None)
        if parent_id:
            children_map[parent_id].append(oc)
    for pid in children_map:
        children_map[pid].sort(key=lambda n: getattr(n, "cdate", 0))

    review_txt_paths: List[str] = []
    submission_discussion_path: Optional[str] = None

    submission_roots = children_map.get(submission.id, [])
    if submission_roots:
        submission_discussion_path = os.path.join(paper_dir, "submission_discussion.txt")
        with open(submission_discussion_path, "w", encoding="utf-8") as out:
            out.write(f"Paper #{paper_number}: {title}\n")
            out.write(f"Forum ID: {submission.id}\n")
            out.write(f"Venue ID: {venue_id}\n")
            out.write(f"Exported: {datetime.utcnow().isoformat()}Z\n")
            out.write("=" * 80 + "\n")
            out.write("SUBMISSION-LEVEL DISCUSSION (Notes not attached to a specific review)\n")
            out.write("-" * 80 + "\n")
            for i, root in enumerate(submission_roots, start=1):
                _write_discussion_thread(
                    out=out,
                    node=root,
                    children_map=children_map,
                    authors_group=authors_group,
                    reviewers_group=reviewers_group,
                    venue_id=venue_id,
                    submission=submission,
                    depth=0,
                    index_path=str(i),
                )

    if include_meta_review and meta_review_mode == "per_paper_file":
        mr_path = os.path.join(paper_dir, "meta_review.txt")
        with open(mr_path, "w", encoding="utf-8") as out:
            out.write(f"Paper #{paper_number}: {title}\n")
            out.write(f"Forum ID: {submission.id}\n")
            out.write(f"Venue ID: {venue_id}\n")
            out.write(f"Exported: {datetime.utcnow().isoformat()}Z\n")
            out.write("=" * 80 + "\n")

            if meta_entries:
                for idx, (note, kind) in enumerate(meta_entries, start=1):
                    header = "META-REVIEW" if kind == "meta" else "PAPER DECISION"
                    out.write(f"SECTION {idx} — {header}\n")
                    out.write("-" * 80 + "\n")
                    if kind == "meta":
                        _render_meta_review(out, note)
                    else:
                        _render_decision_note(out, note)
                    out.write("\n")
            else:
                out.write("(No meta-review or decision note found)\n")

        meta_review_paths.append(mr_path)

    for idx, rev in enumerate(sorted(reviews, key=lambda n: getattr(n, "cdate", 0)), start=1):
        roots = list(children_map.get(rev.id, []))

        out_path = os.path.join(paper_dir, f"review_{idx}.txt")
        with open(out_path, "w", encoding="utf-8") as out:
            out.write(f"Paper #{paper_number}: {title}\n")
            out.write(f"Forum ID: {submission.id}\n")
            out.write(f"Venue ID: {venue_id}\n")
            out.write(f"Exported: {datetime.utcnow().isoformat()}Z\n")
            out.write("=" * 80 + "\n")

            if include_meta_review and meta_review_mode == "embed_in_each_review":
                out.write("SECTION 0 — META-REVIEW(S) / DECISION NOTE(S)\n")
                out.write("-" * 80 + "\n")
                if not meta_entries:
                    out.write("(No meta-review or decision note found)\n\n")
                else:
                    for note, kind in meta_entries:
                        if kind == "meta":
                            _render_meta_review(out, note)
                        else:
                            _render_decision_note(out, note)

            out.write("SECTION 1 — OFFICIAL REVIEW\n")
            out.write("-" * 80 + "\n")
            out.write(f"Signatures: {', '.join(getattr(rev, 'signatures', []) or [])}\n")
            out.write(f"Posted: { _fmt_ts(getattr(rev, 'cdate', 0)) }\n\n")
            _render_review_sections(out, rev, ordered_schema_keys=None)

            out.write("SECTION 2 — DISCUSSION (Official_Comments, multi-round)\n")
            out.write("-" * 80 + "\n")
            if not roots:
                out.write("(No official comments under this review)\n\n")
            else:
                for i, root in enumerate(roots, start=1):
                    _write_discussion_thread(
                        out=out,
                        node=root,
                        children_map=children_map,
                        authors_group=authors_group,
                        reviewers_group=reviewers_group,
                        venue_id=venue_id,
                        submission=submission,
                        depth=0,
                        index_path=str(i),
                    )

            if submission_roots:
                out.write("SECTION 3 — SUBMISSION-LEVEL DISCUSSION\n")
                out.write("-" * 80 + "\n")
                for i, root in enumerate(submission_roots, start=1):
                    _write_discussion_thread(
                        out=out,
                        node=root,
                        children_map=children_map,
                        authors_group=authors_group,
                        reviewers_group=reviewers_group,
                        venue_id=venue_id,
                        submission=submission,
                        depth=0,
                        index_path=str(i),
                    )
                out.write("\n")

        review_txt_paths.append(out_path)

    return {
        "paper_dir": paper_dir,
        "paper_pdf_path": paper_pdf_path,
        "review_txt_paths": review_txt_paths,
        "meta_review_paths": meta_review_paths,
        "submission_discussion_path": submission_discussion_path,
        "paper_number": paper_number,
        "title": title,
        "venue_id": venue_id,
        "forum_id": forum_id,
    }
