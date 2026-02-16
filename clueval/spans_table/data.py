from dataclasses import dataclass


@dataclass
class UnifiedSpan:
    position_start: int
    position_end: int
    token_id_start: int
    token_id_end: int
    text: str
    label: str | list[str]
    doc_id: str
    domain: str


@dataclass
class ParsedSpan:
    position_start: int
    position_end: int
    doc_id: str | None
    head: int | str


@dataclass
class SpanComponent:
    position_start: int
    position_end: int
    doc_id: str | int


@dataclass
class Token:
    position: int
    token_id: int | str | None
    token: str
    label: str | list[str]
    doc_id: str | None
    domain: str | None
