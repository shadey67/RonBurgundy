from typing import Literal
from pydantic import BaseModel, Field

class Profile(BaseModel):
    topics: list[str] = Field(min_length=1, max_length=8, description="Subjects the user wants news about")
    favoured_sources: list[str] = []
    blocked_sources: list[str] = []
    tone: Literal["neutral", "casual", "analytical"]
    max_stories: int = Field(ge=3, le=15)