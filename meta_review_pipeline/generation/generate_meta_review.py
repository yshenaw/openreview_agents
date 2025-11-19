#!/usr/bin/env python3
"""Utilities for generating meta-reviews using configurable LLM providers."""

import argparse
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..utils.api_client import AzureOpenAIClient, OpenAIClient
from ..utils.export_process import extract_submission_data, read_text_file


def generate_meta_review_prompt(
    scores: Optional[str] = None,
    mode: str = "balanced",
    has_confidential_note: bool = False,
    no_rebuttal: bool = False,
) -> str:
    """Generate meta-review prompt text based on the requested mode."""

    score_line = (
        scores)

    preface = ""
    if no_rebuttal:
        preface = (
            "IMPORTANT: The authors have not submitted a rebuttal yet. "
            "Base your decision on whether the paper itself addresses the reviewers' weaknesses and questions.\n\n"
        )

    base_prompt = preface + f"""You are a Area Chair for ICLR 2026, one of the top AI conferences. Your task is to write a comprehensive meta-review based on the paper and all reviewer comments. You don’t need please anybody or use soft words, ICLR is a top-tier conference and thus you should apply high standards in your evaluation.
    If at least one reviewer has given a negative score and you wish to recommend acceptance, please ensure that their major concerns have been adequately addressed in the authors’ response.

ICLR 2026 ACCEPTANCE CATEGORIES:
- **Oral (Top 1-2%)**: Exceptional papers with groundbreaking contributions, flawless execution, and significant impact
- **Spotlight (Top 5%)**: High-quality papers with strong contributions, solid methodology, and clear significance  
- **Poster (Top 25%)**: Good papers with reasonable contributions, adequate execution, and some limitations
- **Reject**: Papers with significant flaws, insufficient novelty, poor execution, or limited impact

The followings are Area Chair Guidelines to write meta-reviews:
- {score_line}
- Don't focus too much on the scores. Instead, look carefully at the comments. Judge the quality of the review rather than taking note of the reviewer's confidence score; the latter may be more a measure of personality.
- Indicate that you have read the author response, even if you just say "the rebuttal did not overcome the reviewer's objections."
- If you use information that is not in the reviews (e.g., from corresponding with one of the reviewers after the rebuttal period), tell the authors (a) that you have done so and (b) what that information is.
- If you find yourself wanting to overrule a unanimous opinion of the referees, the standards for your summary should be at the level of a full review. In these cases, it would probably be best to solicit an auxiliary review.
- Please attempt to take a decisive stand on borderline papers. Other than papers where there is a genuine disagreement, much of our work will involve borderline papers where no one confidently expresses excitement, nor are any major problems identified. These are the tough decisions where we need your judgment!
- Try to counter biases you perceive in the reviews. Unfashionable subjects should be treated fairly but often aren't, to the advantage of the papers on more mainstream approaches. To help the NeurIPS community move faster out of local minima, it is important to encourage risk and recognize that new approaches can't initially yield state-of-the-art competitive results. Nor are they always sold according to the recipes we are used to.

Please provide:

1. **META-REVIEW** following the above guidelines.
2. **FINAL RECOMMENDATION**: One of [Oral, Spotlight, Poster, Reject] with detailed justification, placed at the very end of your response.

Be thorough, fair, and provide specific examples from the paper. Your recommendation should be well-justified based on ICLR standards. Ensure the final recommendation section comes after the meta-review content."""

    if has_confidential_note:
        base_prompt += (
            "\n\nA confidential author-to-AC note is provided. Treat it as potentially subjective context and do not over-rely on it unless corroborated by reviewer feedback or the paper."
        )

    if mode == "strict":
        additional = "\n\nMode: STRICT - Apply high standards. Be conservative with positive recommendations and highlight any significant concerns."
    elif mode == "detailed":
        additional = "\n\nMode: DETAILED - Provide extensive analysis with specific page references and detailed technical commentary."
    else:
        additional = "\n\nMode: BALANCED - Provide fair assessment considering both strengths and weaknesses equally."

    return base_prompt + additional


class MetaReviewGenerator:
    """Generate meta-reviews using the configured LLM provider."""

    def __init__(self, api: str = "azure") -> None:
        api_normalized = api.lower()
        if api_normalized == "azure":
            self.provider = "azure"
            self.client = AzureOpenAIClient()
        elif api_normalized in {"openai", "openai-url", "openai_url", "pdf-url"}:
            self.provider = "openai"
            self.client = OpenAIClient()
        else:
            raise ValueError(f"Unsupported generation API: {api}")

    def generate_meta_review(
        self,
        paper_images: List[Dict],
        reviews: Dict[str, str],
        mode: str = "balanced",
        score_statement: Optional[str] = None,
        confidential_note: Optional[str] = None,
        forum_id: Optional[str] = None,
        paper_pdf: Optional[str] = None,
        no_rebuttal: bool = False,
    ) -> str:
        """Generate a meta-review using the configured LLM provider."""

        prompt = generate_meta_review_prompt(
            scores=score_statement,
            mode=mode,
            has_confidential_note=bool(confidential_note and confidential_note.strip()),
            no_rebuttal=no_rebuttal,
        )

        reviews_text = "\n\n=== REVIEWER COMMENTS ===\n"
        for review_num, review_content in reviews.items():
            reviews_text += f"\n--- Review {review_num} ---\n{review_content}\n"

        print(f"Generating meta-review using {mode} mode via {self.provider} API...")

        try:
            if self.provider == "azure":
                content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
                content.extend(paper_images)

                reviews_block = reviews_text
                if confidential_note and confidential_note.strip():
                    reviews_block += (
                        "\n\n=== CONFIDENTIAL AUTHOR → AREA CHAIR NOTE (Subjective; use cautiously) ===\n"
                        f"{confidential_note.strip()}"
                    )
                content.append(
                    {
                        "type": "text",
                        "text": reviews_block
                        + "\n\nPlease provide your meta-review following the structure above:",
                    }
                )

                return self.client.chat_completion(
                    model=self.client.generation_deployment,
                    messages=[{"role": "user", "content": content}],
                )

            # openai provider path
            pdf_url = None
            if forum_id:
                pdf_url = f"https://openreview.net/pdf?id={forum_id}"
            elif paper_pdf:
                stem = os.path.splitext(os.path.basename(paper_pdf))[0]
                if "_" in stem:
                    candidate = stem.split("_", 1)[0]
                    if candidate:
                        pdf_url = f"https://openreview.net/pdf?id={candidate}"

            if not pdf_url:
                raise ValueError("Unable to determine OpenReview PDF URL for this submission.")

            return self.client.generate_with_pdf(
                prompt_text=prompt,
                reviews_text=reviews_text,
                pdf_url=pdf_url,
                confidential_note=confidential_note,
            )

        except Exception as exc:  # pragma: no cover - network interaction
            return f"Error during meta-review generation: {exc}"


def save_meta_review(
    submission_id: str,
    meta_review_content: str,
    submission_data: Dict[Any, Any],
    mode: str,
    output_path: Optional[str] = None,
) -> str:
    """Persist a generated meta-review report to disk."""

    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"generated_meta_review_{submission_id}_{mode}_{timestamp}.txt"

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("=" * 80 + "\n")
        handle.write("ICLR 2026 GENERATED META-REVIEW\n")
        handle.write("=" * 80 + "\n")
        handle.write(f"Submission ID: {submission_id}\n")
        handle.write(f"Generation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        handle.write(f"Generation Mode: {mode.upper()}\n")
        handle.write(f"Paper PDF: {os.path.basename(submission_data.get('paper_pdf', 'N/A'))}\n")
        handle.write(f"Number of Reviews: {len(submission_data.get('reviews', {}))}\n")
        handle.write(f"Paper Title: {submission_data.get('submission_id', 'N/A')}\n")
        handle.write("\n" + "=" * 80 + "\n")
        handle.write("GENERATED META-REVIEW\n")
        handle.write("=" * 80 + "\n\n")
        handle.write(meta_review_content)
        handle.write("\n\n" + "=" * 80 + "\n")
        handle.write("ORIGINAL REVIEWS SUMMARY\n")
        handle.write("=" * 80 + "\n")

        for review_num, review_content in submission_data.get("reviews", {}).items():
            handle.write(f"\n--- Review {review_num} (Length: {len(review_content)} chars) ---\n")
            preview = review_content[:300] + "..." if len(review_content) > 300 else review_content
            handle.write(preview + "\n")

    return output_path


def extract_recommendation(meta_review_text: str) -> str:
    """Extract the final recommendation token from a generated meta-review."""

    recommendation_map = {
        "oral": "Oral",
        "spotlight": "Spotlight",
        "poster": "Poster",
        "reject": "Reject",
    }

    patterns = [
        r"final\s*recommendation\**\s*[:\-–]?\s*\**\s*(oral|spotlight|poster|reject)\b",
        r"recommendation\**\s*[:\-–]?\s*\**\s*(oral|spotlight|poster|reject)\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, meta_review_text, flags=re.IGNORECASE)
        if match:
            token = match.group(1).strip().lower()
            return recommendation_map.get(token, "Unknown")

    return "Unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ICLR 2026 meta-reviews with Azure OpenAI")
    parser.add_argument("submission_folder", help="Path to submission folder")
    parser.add_argument(
        "--mode",
        choices=["strict", "balanced", "detailed"],
        default="balanced",
        help="Meta-review generation mode (default: balanced)",
    )
    parser.add_argument(
        "--api",
        choices=["azure", "openai"],
        default="azure",
        help="LLM provider to use (default: azure)",
    )
    parser.add_argument("--output", help="Output file path (default: auto-generated)")
    parser.add_argument(
        "--no-rebuttal",
        action="store_true",
        help=(
            "Flag indicating the rebuttal period has not happened; "
            "adds guidance to the prompt to judge without author responses."
        ),
    )

    args = parser.parse_args()

    if not os.path.exists(args.submission_folder):
        raise FileNotFoundError(f"Submission folder not found: {args.submission_folder}")

    print("=" * 80)
    print("ICLR 2026 META-REVIEW GENERATOR")
    print("=" * 80)
    print(f"Submission Folder: {args.submission_folder}")
    print(f"Generation Mode: {args.mode.upper()}")
    print("=" * 80)

    submission_data = extract_submission_data(
        args.submission_folder,
        convert_to_images=(args.api == "azure"),
    )
    if not submission_data["reviews"]:
        raise RuntimeError("No reviews found in submission folder")

    print(f"\n1. Extracted {len(submission_data['reviews'])} reviews")
    print(f"   Paper pages: {len(submission_data['paper_images'])}")

    submission_discussion_path = os.path.join(args.submission_folder, "submission_discussion.txt")
    confidential_note = None
    if os.path.exists(submission_discussion_path):
        confidential_note = read_text_file(submission_discussion_path)
        print("   Found submission_discussion.txt (including as confidential author note).")

    generator = MetaReviewGenerator(api=args.api)

    print(f"\n2. Generating meta-review with {args.api} API...")
    meta_review_content = generator.generate_meta_review(
    paper_images=submission_data["paper_images"],
        reviews=submission_data["reviews"],
        mode=args.mode,
        score_statement=submission_data.get("score_statement"),
        confidential_note=confidential_note,
        forum_id=submission_data.get("forum_id"),
        paper_pdf=submission_data.get("paper_pdf"),
        no_rebuttal=args.no_rebuttal,
    )

    recommendation = extract_recommendation(meta_review_content)

    print("\n3. Saving generated meta-review...")
    output_path = save_meta_review(
        submission_data["submission_id"],
        meta_review_content,
        submission_data,
        args.mode,
        args.output,
    )

    print("\n" + "=" * 80)
    print("META-REVIEW GENERATION COMPLETED")
    print("=" * 80)
    print(f"Output saved to: {output_path}")
    print(f"Recommendation: {recommendation}")


if __name__ == "__main__":
    main()
