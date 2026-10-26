
"""Pydantic schema required by the mRNA-VIP project PRD."""

from typing import List

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ClinicalSafetyReport(BaseModel):
    """Structured educational report; not real clinical guidance."""

    model_config = ConfigDict(extra="forbid")

    Eligible_For_Vaccine: bool
    Recommended_Dose_mg: float = Field(ge=0)
    Reasoning_Steps: List[str]
    FDA_Safety_Warnings_Cited: List[str]
    Disclaimer: str = Field(
        default="Student engineering project output — not real clinical guidance.",
        min_length=1,
    )

    @field_validator("Disclaimer")
    @classmethod
    def disclaimer_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Disclaimer must not be empty.")
        return value
