from pydantic import BaseModel, Field
from typing import Literal


class Parameter(BaseModel):
    type: Literal["string", "integer", "number", "boolean"]


class FunctionDefinition(BaseModel):
    name: str = Field(min_length=1)
    description: str
    parameters: dict[str, Parameter]


class PromptItem(BaseModel):
    prompt: str = Field(min_length=1)


class FunctionCallResult(BaseModel):
    prompt: str
    name: str
    parameters: dict[str, str | int | float | bool]
