"""Helpers for retrieving OpenReview assignments for various roles."""

import re
from typing import Dict, Iterable, List, Optional, Set

def _safe_title(note) -> str:
    title_field = (note.content or {}).get("title", {})
    if isinstance(title_field, dict):
        return title_field.get("value", "")
    if isinstance(title_field, str):
        return title_field
    return ""


def _note_entry(note) -> Dict[str, object]:
    return {
        "number": note.number,
        "forum_id": getattr(note, "forum", None) or note.id,
        "title": _safe_title(note),
    }


def _profile_ids(client) -> Set[str]:
    identifiers: Set[str] = set()
    profile = getattr(client, "profile", None)
    if not profile:
        return identifiers

    primary = getattr(profile, "id", None)
    if primary:
        identifiers.add(str(primary))

    for key in ("ids", "emails"):
        values = getattr(profile, key, [])
        if isinstance(values, (list, tuple, set)):
            identifiers.update(str(v) for v in values if v)
        elif isinstance(values, str):
            identifiers.add(values)

    content = getattr(profile, "content", {}) or {}
    if isinstance(content, dict):
        preferred = content.get("preferred_id")
        if preferred:
            identifiers.add(str(preferred))

    return {identifier for identifier in identifiers if identifier}


def _assignments(client, invitation: str, tails: Iterable[str]) -> List[Dict[str, object]]:
    seen: Dict[str, Dict[str, object]] = {}
    for tail in tails:
        if not tail:
            continue
        edges = client.get_all_edges(invitation=invitation, tail=tail)
        for edge in edges:
            note = client.get_note(edge.head)
            seen[note.id] = _note_entry(note)
    return sorted(seen.values(), key=lambda entry: entry["number"])

def _group_based_assignments(
    client,
    venue_id: str,
    identifiers: Iterable[str],
    pattern: "re.Pattern[str]",
) -> List[Dict[str, object]]:
    """Return submission entries by scanning venue groups matching the provided pattern."""

    prefix = f"{venue_id}/Submission"
    seen_numbers: Set[int] = set()
    results: List[Dict[str, object]] = []

    for identifier in identifiers:
        try:
            groups = client.get_groups(prefix=prefix, member=identifier)
        except Exception:  # pragma: no cover - depends on OpenReview permissions
            continue

        for group in groups:
            gid = getattr(group, "id", "")
            match = pattern.search(gid)
            if not match:
                continue

            try:
                number = int(match.group(1))
            except (IndexError, ValueError):
                continue

            if number in seen_numbers:
                continue

            note_obj = None
            try:
                note_obj = client.get_note_by_number(venue_id, number)
            except Exception:  # pragma: no cover - depends on OpenReview permissions
                note_obj = None

            if note_obj is None:
                try:
                    candidates = client.get_all_notes(
                        invitation=f"{venue_id}/-/Submission",
                        number=number,
                    )
                except Exception:  # pragma: no cover - depends on OpenReview permissions
                    candidates = []

                if candidates:
                    note_obj = candidates[0]

            if note_obj is None:
                continue

            results.append(_note_entry(note_obj))
            seen_numbers.add(number)

    return sorted(results, key=lambda entry: entry["number"])


def _matches_paper_type(note, target_label: Optional[str]) -> bool:
    if not target_label:
        return True
    target = target_label.lower()
    synonyms = {
        "oral": ("oral",),
        "spotlight": ("spotlight",),
        "poster": ("poster",),
        "rejected": ("rejected", "withdrawn", "submitted"),
    }
    terms = synonyms.get(target, (target,))

    venue_field = (note.content or {}).get("venue", {})
    venue_value: Optional[str] = None
    if isinstance(venue_field, dict):
        venue_value = venue_field.get("value")
    elif isinstance(venue_field, str):
        venue_value = venue_field

    lower_venue = venue_value.lower() if isinstance(venue_value, str) else ""
    if any(term in lower_venue for term in terms):
        return True

    venueid_field = (note.content or {}).get("venueid", {})
    if isinstance(venueid_field, dict):
        venueid_value = venueid_field.get("value")
    elif isinstance(venueid_field, str):
        venueid_value = venueid_field
    else:
        venueid_value = None

    lower_venueid = venueid_value.lower() if isinstance(venueid_value, str) else ""
    if any(term in lower_venueid for term in terms):
        return True

    return False


def _note_entry_for_forum(client, forum_id: str):
    try:
        note = client.get_note(forum_id)
        if note is not None:
            return note
    except Exception:
        note = None

    try:
        notes = client.get_all_notes(forum=forum_id)
    except Exception:
        notes = []

    for candidate in notes:
        if getattr(candidate, "id", None) == forum_id or getattr(candidate, "forum", None) == forum_id:
            return candidate

    return None


def get_papers_audience(
    client,
    venue_id: str,
    paper_type: Optional[str] = None,
    forum_ids: Optional[Iterable[str]] = None,
) -> List[Dict[str, object]]:
    """Return public submissions matching the requested acceptance type for audience users."""

    invitation = f"{venue_id}/-/Submission"
    target = paper_type.lower() if paper_type else None
    if target and target not in {"oral", "spotlight", "poster", "rejected"}:
        raise ValueError("paper_type must be one of: oral, spotlight, poster, rejected")

    normalized_ids: List[str] = []
    if forum_ids:
        for raw in forum_ids:
            if not raw:
                continue
            parts = [segment.strip() for segment in str(raw).split(",")]
            normalized_ids.extend([part for part in parts if part])

    if normalized_ids:
        notes = []
        seen: Set[str] = set()
        for forum_id in normalized_ids:
            if forum_id in seen:
                continue
            seen.add(forum_id)
            note_obj = _note_entry_for_forum(client, forum_id)
            if note_obj is None:
                continue
            if not _matches_paper_type(note_obj, target):
                continue
            notes.append(note_obj)
    else:
        try:
            notes = client.get_all_notes(invitation=invitation)
        except Exception:  # pragma: no cover - depends on OpenReview permissions
            notes = []

    results = [
        _note_entry(note)
        for note in notes
        if _matches_paper_type(note, target)
    ]

    return sorted(results, key=lambda entry: entry["number"])


def get_papers(client, venue_id):
    """Return submissions assigned to the caller as Area Chair."""
    ac_inv = f"{venue_id}/Area_Chairs/-/Assignment"
    tails = _profile_ids(client) or {client.profile.id}
    return _assignments(client, ac_inv, tails)


def get_papers_sac(client, venue_id):
    """Return submissions assigned to the caller as Senior Area Chair."""
    sac_inv = f"{venue_id}/Senior_Area_Chairs/-/Assignment"
    tails = _profile_ids(client) or {client.profile.id}
    return _assignments(client, sac_inv, tails)


def get_papers_reviewer(client, venue_id):
    """Return submissions assigned to the caller as Reviewer."""
    identifiers = _profile_ids(client)
    profile = getattr(client, "profile", None)
    if not identifiers and profile is not None:
        primary = getattr(profile, "id", None)
        if primary:
            identifiers = {str(primary)}

    reviewer_pattern = re.compile(r"/Submission(\d+)/Reviewer_", re.IGNORECASE)
    return _group_based_assignments(client, venue_id, identifiers, reviewer_pattern)


def get_papers_author(client, venue_id):
    """Return submissions where the caller appears in the Authors group."""
    identifiers = _profile_ids(client)
    profile = getattr(client, "profile", None)
    if not identifiers and profile is not None:
        primary = getattr(profile, "id", None)
        if primary:
            identifiers = {str(primary)}

    author_pattern = re.compile(r"/Submission(\d+)/Authors", re.IGNORECASE)
    return _group_based_assignments(client, venue_id, identifiers, author_pattern)

