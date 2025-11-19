"""LLM client utilities used by the meta-review pipeline."""

import os
from typing import Any, Dict, List, Optional

from azure.identity import (
    AzureCliCredential,
    ChainedTokenCredential,
    ManagedIdentityCredential,
    get_bearer_token_provider,
)
from openai import AzureOpenAI, OpenAI


class AzureOpenAIClient:
    """Azure OpenAI client wrapper configured for the meta-review workflow."""

    def __init__(self) -> None:
        # Azure OpenAI configuration
        self.scope = "api://trapi/.default"
        self.credential = get_bearer_token_provider(
            ChainedTokenCredential(
                AzureCliCredential(),
                ManagedIdentityCredential(),
            ),
            self.scope,
        )

        self.api_version = "2024-12-01-preview"
        self.generation_deployment = "gpt-5_2025-08-07"#"gpt-5_2025-08-07"
        self.evaluation_deployment = "gpt-4o_2024-11-20"#"gpt-4o_2024-11-20"
        self.instance = "msra/shared"
        self.endpoint = f"https://trapi.research.microsoft.com/{self.instance}"

        self.client = AzureOpenAI(
            azure_endpoint=self.endpoint,
            azure_ad_token_provider=self.credential,
            api_version=self.api_version,
        )

    def chat_completion(self, *, model: str, messages: List[Dict[str, Any]], **kwargs: Any) -> str:
        """Call Azure OpenAI chat completions and return the first message content."""

        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs,
        )

        return response.choices[0].message.content


class OpenAIClient:
    """OpenAI client that supports PDF URLs directly via the Responses API."""

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        generation_model: Optional[str] = None,
        evaluation_model: Optional[str] = None,
    ) -> None:
        self.base_url = base_url or os.environ.get("OPENAI_PDF_BASE_URL", "https://api.deerapi.com/v1")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable must be set for PDF URL client.")

        self.client = OpenAI(api_key=self.api_key) #base_url=self.base_url, 
        self.generation_model = generation_model or os.environ.get("OPENAI_PDF_GENERATION_MODEL", "gpt-5")
        self.evaluation_model = evaluation_model or os.environ.get("OPENAI_PDF_EVALUATION_MODEL", "gpt-4o")

    def generate_with_pdf(
        self,
        *,
        prompt_text: str,
        reviews_text: str,
        pdf_url: str,
        confidential_note: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        content_text = prompt_text
        if reviews_text:
            content_text += "\n\n" + reviews_text
        if confidential_note:
            content_text += (
                "\n\n=== CONFIDENTIAL AUTHOR → AREA CHAIR NOTE (Subjective; use cautiously) ===\n"
                + confidential_note.strip()
            )
        content_text += "\n\nPlease provide your meta-review following the requested structure."

        response = self.client.responses.create(
            model=self.generation_model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": content_text},
                        {"type": "input_file", "file_url": pdf_url},
                    ],
                }
            ],
            **kwargs,
        )
        return response.output_text

    def evaluate_with_text(self, *, prompt_text: str, **kwargs: Any) -> str:
        response = self.client.responses.create(
            model=self.evaluation_model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt_text},
                    ],
                }
            ],
            **kwargs,
        )
        return response.output_text
