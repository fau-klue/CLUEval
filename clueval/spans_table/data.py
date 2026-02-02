from dataclasses import dataclass


@dataclass
class UnifiedSpan:
    start_id: int
    end_id: int
    doc_token_id_start: int
    doc_token_id_end: int
    text: str
    label: str | list[str]
    doc_id: str
    domain: str


@dataclass
class ParsedSpan:
    start_id: int
    end_id: int
    doc_id: str | None
    head: int | str


@dataclass
class SpanComponent:
    start_id: int
    end_id: int
    doc_id: str | int


@dataclass
class Token:
    position: int
    token_id: int | str | None
    token: str
    label: str | list[str]
    doc_id: str | None
    domain: str | None
