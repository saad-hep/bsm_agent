"""Interactive chat CLI for BSM model-building workflows."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
import json
from pathlib import Path
import re
import threading
import time
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from .agent_backend import build_model_from_payload, build_report_from_payload
from .fields import field_latex_name
from .groups import SU3Rep
from .remote_api import build_chat_model, resolve_model_target


SYSTEM_PROMPT = """You are a BSM model-building assistant.

You help the user build Standard Model extensions and generate compact reports.

Operational rules:
- When the user asks to construct or analyze a specific BSM field content, call `build_model`.
- When the user asks for a PDF, TeX report, written output files, or a saved artifact, call `build_report`.
- Before any nontrivial BSM build or report generation, first interpret the field content clearly.
- Use exact JSON-like tool arguments. `fields` must be a list of objects with:
  name, kind, su3, su2, hypercharge, and real.
- For `build_report`, default to a compact report unless the user explicitly asks to add expanded Lagrangian sections or mass matrices.
- Valid `expand_sectors` values are subsets of: `yukawa`, `scalar_potential`, `gauge_kinetic`.
- Keep conceptual answers short and technically precise.
- Do not invent generated terms or anomaly results. Use tool outputs for those.
- In CLI responses, prefer plain text over Markdown. Do not use bold markers, nested bullets, or LaTeX fragments unless explicitly asked.
"""

FIELD_INTERPRETATION_PROMPT = """Interpret the user's message as candidate BSM field content.

Rules for this extraction step:
- If the message contains enough information to specify one or more fields, call `build_model`.
- Use the minimal field objects needed: `su3`, `su2`, `hypercharge`, and only include `kind`, `name`, or `real` when the user explicitly provided them or they are strictly necessary.
- Do not ask for a field name if none was given. Do not ask for generations if none were given.
- If the user did not specify scalar vs fermion, omit `kind` and let the confirmation step ask that single follow-up.
- If the message is too incomplete to identify the field content, ask only for the missing quantum numbers.
- If you call a tool, use strict JSON arguments.
"""

CURRENT_MODEL_ADDITION_PROMPT = """Interpret the user's message as fields to add to the current BSM model.

Rules for this extraction step:
- Extract only the new fields the user wants to add. Do not repeat or modify existing fields.
- If the message contains enough information to specify one or more new fields, call `build_model`.
- Use the minimal field objects needed: `su3`, `su2`, `hypercharge`, and only include `kind`, `name`, or `real` when the user explicitly provided them or they are strictly necessary.
- If the user clearly identified a standard particle label such as a right-handed neutrino, infer its standard SM quantum numbers.
- Do not ask for a field name if none was given. Do not ask for generations if none were given.
- If the user did not specify scalar vs fermion, omit `kind` and let the confirmation step ask that single follow-up.
- If the message is too incomplete to identify the new fields, ask only for the missing quantum numbers.
- If you call a tool, use strict JSON arguments.
"""


@tool
def build_model(
    fields: list[dict[str, Any]] | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Build a BSM model from field specifications and return its summary."""

    payload: dict[str, Any] = {"fields": fields or []}
    if model_name:
        payload["model_name"] = model_name
    result = build_model_from_payload(payload)
    return {
        "model_name": result.model_name,
        "fields": [field.__dict__ for field in result.fields],
        "lagrangian_summary": result.lagrangian_summary,
        "anomaly_summary": result.anomaly_summary,
        "kinetic_term_count": result.kinetic_term_count,
        "yukawa_terms": list(result.yukawa_terms),
        "fermion_mass_terms": list(result.fermion_mass_terms),
        "scalar_potential_terms": list(result.scalar_potential_terms),
        "mixed_bsm_terms": list(result.mixed_bsm_terms),
    }


@tool
def build_report(
    fields: list[dict[str, Any]] | None = None,
    model_name: str | None = None,
    output_stem: str | None = None,
    output_dir: str = "output",
    make_pdf: bool = True,
    expand_sectors: list[str] | str | None = None,
    ewsb: bool | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a TeX report and optional PDF for a BSM model."""

    payload: dict[str, Any] = {
        "fields": fields or [],
        "output_dir": output_dir,
        "make_pdf": make_pdf,
    }
    if expand_sectors is not None:
        payload["expand_sectors"] = expand_sectors
    if ewsb is not None:
        payload["ewsb"] = ewsb
    if model_name:
        payload["model_name"] = model_name
    if output_stem:
        payload["output_stem"] = output_stem
    result = build_report_from_payload(payload, base_dir=Path.cwd())
    return {
        "model_name": result.model_name,
        "output_stem": result.output_stem,
        "output_dir": str(result.output_dir),
        "tex_path": str(result.tex_path),
        "pdf_path": str(result.pdf_path) if result.pdf_path else None,
        "lagrangian_summary": result.lagrangian_summary,
        "anomaly_summary": result.anomaly_summary,
        "mixed_bsm_terms": list(result.mixed_bsm_terms),
        "fields": [field.__dict__ for field in result.fields],
    }


TOOLS = [build_model, build_report]
TOOL_MAP = {tool_obj.name: tool_obj for tool_obj in TOOLS}

_SU2_ALIASES = {
    "singlet": 1,
    "doublet": 2,
    "triplet": 3,
    "quartet": 4,
    "quintet": 5,
    "sextet": 6,
    "septet": 7,
}

_FRACTION_WORD_ALIASES = {
    "zero": "0",
    "one": "1",
    "minus one": "-1",
    "negative one": "-1",
    "two": "2",
    "minus two": "-2",
    "negative two": "-2",
    "one half": "1/2",
    "minus one half": "-1/2",
    "negative one half": "-1/2",
    "one third": "1/3",
    "minus one third": "-1/3",
    "negative one third": "-1/3",
    "two third": "2/3",
    "two thirds": "2/3",
    "minus two third": "-2/3",
    "minus two thirds": "-2/3",
    "negative two third": "-2/3",
    "negative two thirds": "-2/3",
    "four third": "4/3",
    "four thirds": "4/3",
    "minus four third": "-4/3",
    "minus four thirds": "-4/3",
    "negative four third": "-4/3",
    "negative four thirds": "-4/3",
    "one sixth": "1/6",
    "minus one sixth": "-1/6",
    "negative one sixth": "-1/6",
    "five sixth": "5/6",
    "five sixths": "5/6",
    "minus five sixth": "-5/6",
    "minus five sixths": "-5/6",
    "negative five sixth": "-5/6",
    "negative five sixths": "-5/6",
}

_SU3_DIRECT_PATTERNS = (
    "anti sextet",
    "antisextet",
    "anti triplet",
    "antitriplet",
    "anti fundamental",
    "antifundamental",
    "decuplet",
    "sextet",
    "octet",
    "triplet",
    "singlet",
)

_SU2_DIRECT_PATTERNS = (
    "doublet",
    "triplet",
    "quartet",
    "quintet",
    "sextet",
    "septet",
    "singlet",
)

_SU3_CONTEXT_TOKENS = ("color", "colour", "su3", "su(3)")
_SU2_CONTEXT_TOKENS = ("weak", "su2", "su(2)")


@dataclass(frozen=True)
class PendingBuildConfirmation:
    tool_name: str
    tool_args: dict[str, Any]
    tool_call_id: str
    requires_kind_confirmation: bool = False
    confirmation_fields: list[dict[str, Any]] | None = None


@dataclass(frozen=True)
class CurrentModelState:
    result: dict[str, Any]


AGENT_STYLE_PREFIX = "\033[1;36m"
AGENT_STYLE_SUFFIX = "\033[0m"
USER_PROMPT = "\033[1;35mUser>\033[0m "
RED_HIGHLIGHT = "\033[1;31m"
GREEN_HIGHLIGHT = "\033[1;32m"


@contextmanager
def _activity_indicator(label: str = "Agent is working"):
    stop_event = threading.Event()

    def spin() -> None:
        frames = "|/-\\"
        index = 0
        while not stop_event.is_set():
            frame = frames[index % len(frames)]
            print(f"\r{label} ... {frame}", end="", flush=True)
            index += 1
            time.sleep(0.12)
        print("\r" + " " * (len(label) + 10) + "\r", end="", flush=True)

    worker = threading.Thread(target=spin, daemon=True)
    worker.start()
    try:
        yield
    finally:
        stop_event.set()
        worker.join()


def _normalize_ai_response(response: Any) -> AIMessage:
    if isinstance(response, AIMessage):
        return response
    if isinstance(response, dict):
        return AIMessage(
            content=response.get("content", ""),
            tool_calls=response.get("tool_calls", []),
        )
    raise TypeError(f"Unsupported LLM response type: {type(response)!r}")


def _tool_result_message(tool_call: dict[str, Any]) -> ToolMessage:
    name = tool_call["name"]
    if name not in TOOL_MAP:
        raise ValueError(f"Unsupported tool call: {name}")
    args = _normalize_tool_args(tool_call.get("args", {}))
    result = TOOL_MAP[name].invoke(args)
    return ToolMessage(
        content=json.dumps({"tool_name": name, **result}, default=str),
        tool_call_id=tool_call["id"],
    )


def _field_signature(field: dict[str, Any]) -> str:
    return f"({field['su3']},{field['su2']},{field['hypercharge']})"


def _format_field_content(fields: list[dict[str, Any]]) -> str:
    return ", ".join(_field_signature(field) for field in fields)


def _format_confirmation_field_content(fields: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for field in fields:
        kind = str(field.get("kind") or "").strip()
        if kind == "scalar" and bool(field.get("real")):
            kind = "real scalar"
        signature = _field_signature(field)
        parts.append(f"{kind} {signature}" if kind else signature)
    return ", ".join(parts)


def _format_agent_output(text: str) -> str:
    return f"{AGENT_STYLE_PREFIX}BSM_Agent> {text}{AGENT_STYLE_SUFFIX}"


def _should_show_step_timing(text: str) -> bool:
    stripped = str(text).strip()
    return stripped.startswith(
        (
            "Generated report ",
            "Generated the Standard Model report.",
            "The model is constructed",
            "Renamed ",
        )
    )


def _append_step_timing(text: str, elapsed_seconds: float) -> str:
    if not _should_show_step_timing(text):
        return text
    return f"{text} [{elapsed_seconds:.1f}s]"


def _highlight_anomalous(text: str) -> str:
    return f"{RED_HIGHLIGHT}{text}{AGENT_STYLE_PREFIX}"


def _highlight_anomaly_free(text: str) -> str:
    return f"{GREEN_HIGHLIGHT}{text}{AGENT_STYLE_PREFIX}"


def _pending_confirmation_fields(pending: PendingBuildConfirmation) -> list[dict[str, Any]]:
    if pending.confirmation_fields is not None:
        return pending.confirmation_fields
    fields = pending.tool_args.get("fields") or []
    return [field for field in fields if isinstance(field, dict)]


def _pending_confirmation_from_tool_calls(
    tool_calls: list[dict[str, Any]],
    *,
    requires_kind_confirmation: bool,
    question: str,
) -> PendingBuildConfirmation | None:
    for tool_call in tool_calls:
        if tool_call["name"] not in {"build_model", "build_report"}:
            continue
        args = _normalize_tool_args(tool_call.get("args", {}))
        args = _canonicalize_field_names_for_tool(args, tool_call["name"], question)
        fields = args.get("fields") or []
        if not fields:
            continue
        missing_kind = any(not isinstance(field, dict) or not str(field.get("kind") or "").strip() for field in fields)
        return PendingBuildConfirmation(
            tool_name=tool_call["name"],
            tool_args=args,
            tool_call_id=tool_call["id"],
            requires_kind_confirmation=requires_kind_confirmation and missing_kind,
        )
    return None


def _normalize_tool_args(args: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(args, dict):
        return args
    normalized = dict(args)
    fields = normalized.get("fields")
    if isinstance(fields, list):
        normalized["fields"] = [_normalize_field_payload(field) for field in fields]
    return normalized


def _canonicalize_field_names_for_tool(args: dict[str, Any], tool_name: str, question: str) -> dict[str, Any]:
    fields = args.get("fields")
    if not isinstance(fields, list):
        return args
    updated_args = dict(args)
    updated_fields: list[Any] = []
    for field in fields:
        if not isinstance(field, dict):
            updated_fields.append(field)
            continue
        updated_field = dict(field)
        name = str(updated_field.get("name") or "").strip()
        if name:
            if tool_name == "build_report" and _question_mentions_field_name(question, name):
                updated_field["latex_name"] = name
            updated_field.pop("name", None)
        updated_fields.append(updated_field)
    updated_args["fields"] = updated_fields
    return updated_args


def _question_mentions_field_name(question: str, name: str) -> bool:
    lowered_question = question.lower()
    lowered_name = name.lower()
    if lowered_name in lowered_question:
        return True
    spaced = lowered_name.replace("_", " ")
    if spaced != lowered_name and spaced in lowered_question:
        return True
    return False


def _looks_like_build_request(question: str) -> bool:
    lowered = " ".join(question.lower().split())
    return any(
        token in lowered
        for token in (
            "construct",
            "build",
            "generate a model",
            "make a model",
            "extend the sm",
            "add to the model",
            "add a",
            "add an",
        )
    )


def _looks_like_field_content_statement(question: str) -> bool:
    lowered = " ".join(question.lower().split())
    normalized_lowered = lowered.replace("-", " ")
    if re.search(r"\(\s*[^,]+\s*,\s*[^,]+\s*,\s*[^)]+\)", question):
        return True
    if re.search(r"(?:^|[\s:])([^,\s()]+)\s*,\s*([^,\s()]+)\s*,\s*([^,\s()]+)", question):
        return True

    has_su3 = _extract_su3_from_text(normalized_lowered) is not None
    has_su2 = _extract_su2_from_text(normalized_lowered) is not None
    has_hypercharge = re.search(r"(?:hypercharge|y)\s*(?:=|of)?\s+", normalized_lowered) is not None
    return has_su3 and has_su2 and has_hypercharge


def _looks_like_field_content_candidate(question: str) -> bool:
    lowered = " ".join(question.lower().split())
    normalized_lowered = lowered.replace("-", " ")
    if _looks_like_build_request(question) or _looks_like_field_content_statement(question):
        return True
    if re.search(r"\(\s*[^,]+\s*,\s*[^,]+\s*,\s*[^)]+\)", question):
        return True
    if re.search(r"(?:^|[\s:])([^,\s()]+)\s*,\s*([^,\s()]+)\s*,\s*([^,\s()]+)", question):
        return True

    su3_cues = (
        "color",
        "colour",
        "su3",
        "su(3)",
        "fundamental",
        "antifundamental",
        "anti-fundamental",
        "triplet",
        "antitriplet",
        "anti-triplet",
        "sextet",
        "octet",
        "decuplet",
        "bar3",
        "bar6",
        "bar10",
    )
    su2_cues = (
        "weak",
        "su2",
        "su(2)",
        "singlet",
        "doublet",
        "triplet",
        "quartet",
        "quintet",
        "sextet",
        "septet",
    )
    has_su3 = any(cue in normalized_lowered for cue in su3_cues)
    has_su2 = any(cue in normalized_lowered for cue in su2_cues)
    has_hypercharge = "hypercharge" in normalized_lowered or re.search(r"\by\s*=", normalized_lowered) is not None
    return has_su3 and has_su2 and has_hypercharge


def _looks_like_multi_field_request(question: str) -> bool:
    lowered = " ".join(question.lower().split())
    if len(re.findall(r"\(\s*[^,()]+\s*,\s*[^,()]+\s*,\s*[^)]+\)", question)) > 1:
        return True
    if re.search(r"\b(one\s+scalar\b.*\bone\s+fermion|one\s+fermion\b.*\bone\s+scalar)\b", lowered):
        return True
    if re.search(r"\bscalar\b", lowered) and re.search(r"\bfermion\b", lowered):
        return True
    if re.search(r"\b(two|three|\d+)\s+(fields|states)\b", lowered):
        return True
    if "both" in lowered and any(token in lowered for token in ("scalar", "fermion", "fields", "states")):
        return True
    if any(phrase in lowered for phrase in ("these bsm states", "these states", "these fields")):
        return True
    return False


def _contains_group_context(text: str, tokens: tuple[str, ...]) -> bool:
    return any(token in text for token in tokens)


def _extract_su3_from_text(normalized_lowered: str) -> Any:
    for pattern in _SU3_DIRECT_PATTERNS:
        escaped = re.escape(pattern)
        if re.search(rf"(?:color|colour)\s+{escaped}\b", normalized_lowered):
            return _normalize_su3_value(pattern.replace(" ", "-"))
        if re.search(rf"{escaped}\s+(?:color|colour)\b", normalized_lowered):
            return _normalize_su3_value(pattern.replace(" ", "-"))
        if re.search(rf"{escaped}\s+under\s+(?:color|colour|su3|su\(3\))(?=[\s,.;:]|$)", normalized_lowered):
            return _normalize_su3_value(pattern.replace(" ", "-"))
        if re.search(rf"{escaped}\s+of\s+(?:color|colour|su3|su\(3\))(?=[\s,.;:]|$)", normalized_lowered):
            return _normalize_su3_value(pattern.replace(" ", "-"))
        if re.search(rf"(?:color|colour|su3|su\(3\))\s+{escaped}\b", normalized_lowered):
            return _normalize_su3_value(pattern.replace(" ", "-"))
        if re.search(rf"(?:color|colour|su3|su\(3\))\s+(?:is\s+)?{escaped}\b", normalized_lowered):
            return _normalize_su3_value(pattern.replace(" ", "-"))
    return None


def _extract_su2_from_text(normalized_lowered: str) -> Any:
    for pattern in _SU2_DIRECT_PATTERNS:
        escaped = re.escape(pattern)
        if re.search(rf"weak\s+{escaped}\b", normalized_lowered):
            return _normalize_su2_value(pattern)
        if re.search(rf"{escaped}\s+of\s+(?:weak|su2|su\(2\))(?=[\s,.;:]|$)", normalized_lowered):
            return _normalize_su2_value(pattern)
        if re.search(rf"{escaped}\s+under\s+(?:weak|su2|su\(2\))(?=[\s,.;:]|$)", normalized_lowered):
            return _normalize_su2_value(pattern)
        if re.search(rf"(?:weak|su2|su\(2\))\s+{escaped}\b", normalized_lowered):
            return _normalize_su2_value(pattern)
        if re.search(rf"(?:weak|su2|su\(2\))\s+(?:is\s+)?{escaped}\b", normalized_lowered):
            return _normalize_su2_value(pattern)
    return None


def _should_apply_global_singlet_shorthand(normalized_lowered: str) -> bool:
    if re.search(r"\bsinglet\b", normalized_lowered) is None:
        return False
    if _contains_group_context(normalized_lowered, _SU3_CONTEXT_TOKENS + _SU2_CONTEXT_TOKENS):
        return False
    if "hypercharge" in normalized_lowered or re.search(r"\by\s*=", normalized_lowered) is not None:
        return False
    return True


def _extract_tuple_fields(question: str) -> list[dict[str, Any]]:
    tuple_matches = list(re.finditer(r"\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^)]+)\)", question))
    if not tuple_matches:
        return []

    prefix_kind = _extract_explicit_field_kind(question[: tuple_matches[0].start()])
    prefix_reality = _extract_explicit_scalar_reality(question[: tuple_matches[0].start()])
    current_kind = prefix_kind
    current_reality = prefix_reality if prefix_kind == "scalar" else None
    last_end = 0
    fields: list[dict[str, Any]] = []

    for match in tuple_matches:
        context = question[last_end : match.start()]
        local_kind = _extract_explicit_field_kind(context)
        if local_kind is not None:
            current_kind = local_kind
            current_reality = _extract_explicit_scalar_reality(context) if local_kind == "scalar" else None
        elif current_kind == "scalar":
            local_reality = _extract_explicit_scalar_reality(context)
            if local_reality is not None:
                current_reality = local_reality

        field: dict[str, Any] = {
            "su3": _normalize_su3_value(match.group(1)),
            "su2": _normalize_su2_value(match.group(2)),
            "hypercharge": _normalize_hypercharge_value(match.group(3)),
            "generations": 1,
        }
        if current_kind is not None:
            field["kind"] = current_kind
            if current_kind == "scalar":
                field["real"] = False if current_reality is None else current_reality
        fields.append(field)
        last_end = match.end()

    return fields


def _extract_direct_field_request(question: str) -> dict[str, Any] | None:
    normalized_lowered = " ".join(question.lower().split()).replace("-", " ")
    shorthand_singlet_request = _extract_explicit_field_kind(question) is not None and _should_apply_global_singlet_shorthand(normalized_lowered)
    if not (_looks_like_build_request(question) or _looks_like_field_content_statement(question) or shorthand_singlet_request):
        return None
    if _looks_like_multi_field_request(question):
        fields = _extract_tuple_fields(question)
        if not fields:
            return None
        return {
            "tool_name": "build_model",
            "tool_args": {"fields": fields},
            "requires_kind_confirmation": any(not str(field.get("kind") or "").strip() for field in fields),
        }

    tuple_match = re.search(r"\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^)]+)\)", question)
    comma_match = re.search(r"(?:^|[\s:])([^,\s()]+)\s*,\s*([^,\s()]+)\s*,\s*([^,\s()]+)", question)
    su3: Any = None
    su2: Any = None
    hypercharge: Any = None

    if tuple_match is not None:
        su3 = _normalize_su3_value(tuple_match.group(1))
        su2 = _normalize_su2_value(tuple_match.group(2))
        hypercharge = _normalize_hypercharge_value(tuple_match.group(3))
    elif comma_match is not None:
        candidate_su3 = _normalize_su3_value(comma_match.group(1))
        candidate_su2 = _normalize_su2_value(comma_match.group(2))
        candidate_hypercharge = _normalize_hypercharge_value(comma_match.group(3))
        if _looks_like_quantum_number_triplet(candidate_su3, candidate_su2, candidate_hypercharge):
            su3 = candidate_su3
            su2 = candidate_su2
            hypercharge = candidate_hypercharge
    if su3 is None or su2 is None or hypercharge is None:
        lowered = " ".join(question.lower().split())
        normalized_lowered = lowered.replace("-", " ")
        if _should_apply_global_singlet_shorthand(normalized_lowered):
            su3 = "1"
            su2 = 1
            hypercharge = "0"
        if su3 is None:
            su3 = _extract_su3_from_text(normalized_lowered)
        if su2 is None:
            su2 = _extract_su2_from_text(normalized_lowered)
        # Preserve minus signs in hypercharge values; the normalized text is only
        # for matching rep keywords like "anti-triplet".
        hyper_match = re.search(r"(?:hypercharge|y)\s*(?:=|of)?\s+(.+?)(?:,| with | and |$)", lowered)
        if hyper_match is not None:
            hypercharge = _normalize_hypercharge_value(hyper_match.group(1))

    if su3 is None or su2 is None or hypercharge is None:
        return None

    kind = _extract_explicit_field_kind(question)
    field: dict[str, Any] = {
        "su3": su3,
        "su2": su2,
        "hypercharge": hypercharge,
        "generations": 1,
    }
    if kind is not None:
        field["kind"] = kind
        if kind == "scalar":
            reality = _extract_explicit_scalar_reality(question)
            field["real"] = False if reality is None else reality
    return {
        "tool_name": "build_model",
        "tool_args": {"fields": [field]},
        "requires_kind_confirmation": kind is None,
    }


def _normalize_field_payload(field: Any) -> Any:
    if not isinstance(field, dict):
        return field
    normalized = dict(field)
    if "su3" in normalized:
        normalized["su3"] = _normalize_su3_value(normalized["su3"])
    if "su2" in normalized:
        normalized["su2"] = _normalize_su2_value(normalized["su2"])
    if "hypercharge" in normalized:
        normalized["hypercharge"] = _normalize_hypercharge_value(normalized["hypercharge"])
    return normalized


def _question_explicitly_mentions_field_kind(question: str) -> bool:
    return _extract_explicit_field_kind(question) is not None


def _extract_explicit_field_kind(text: str) -> str | None:
    lowered = " ".join(re.sub(r"(?<=[a-z])-(?=[a-z])", " ", text.lower()).split())
    matches = re.findall(r"\b(scalar|scalars|fermion|fermions|fermon|fermons)\b", lowered)
    if matches:
        last = matches[-1]
        if last.startswith("scalar"):
            return "scalar"
        return "fermion"
    return None


def _extract_explicit_scalar_reality(text: str) -> bool | None:
    lowered = " ".join(text.lower().split())
    has_scalar = re.search(r"\bscalar\b", lowered) is not None
    has_real = re.search(r"\breal\b", lowered) is not None
    has_complex = re.search(r"\bcomplex\b", lowered) is not None
    if has_scalar and has_real and not has_complex:
        return True
    if has_scalar and has_complex:
        return False
    return None


def _apply_field_kind(tool_args: dict[str, Any], kind: str) -> dict[str, Any]:
    updated_args = dict(tool_args)
    fields = []
    for field in updated_args.get("fields") or []:
        if not isinstance(field, dict):
            fields.append(field)
            continue
        updated_field = dict(field)
        if not str(updated_field.get("kind") or "").strip():
            updated_field["kind"] = kind
            if kind != "scalar":
                updated_field["real"] = False
        fields.append(updated_field)
    updated_args["fields"] = fields
    return updated_args


def _unwrap_singleton_container(value: Any) -> Any:
    current = value
    while isinstance(current, list) and len(current) == 1:
        current = current[0]
    return current


def _normalize_su3_value(value: Any) -> Any:
    value = _unwrap_singleton_container(value)
    text = str(value).strip()
    if not text:
        return value
    try:
        return SU3Rep.parse(text).label()
    except ValueError:
        return text


def _normalize_su2_value(value: Any) -> Any:
    value = _unwrap_singleton_container(value)
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return value
    lowered = text.lower()
    if lowered in _SU2_ALIASES:
        return _SU2_ALIASES[lowered]
    try:
        return int(text)
    except ValueError:
        return value


def _normalize_hypercharge_value(value: Any) -> Any:
    value = _unwrap_singleton_container(value)
    text = str(value).strip().rstrip(".,;:")
    if not text:
        return value
    lowered = " ".join(re.sub(r"(?<=[a-z])-(?=[a-z])", " ", text.lower()).split())
    lowered = _FRACTION_WORD_ALIASES.get(lowered, lowered)
    if lowered.startswith("minus "):
        lowered = f"-{lowered.removeprefix('minus ').strip()}"
    elif lowered.startswith("negative "):
        lowered = f"-{lowered.removeprefix('negative ').strip()}"
    try:
        return str(Fraction(lowered).limit_denominator(1000))
    except (ValueError, ZeroDivisionError):
        return value



def _is_valid_su3_value(value: Any) -> bool:
    try:
        SU3Rep.parse(str(value).strip())
    except ValueError:
        return False
    return True


def _is_valid_su2_value(value: Any) -> bool:
    return isinstance(value, int) and value >= 1


def _is_valid_hypercharge_value(value: Any) -> bool:
    try:
        Fraction(str(value).strip())
    except (ValueError, ZeroDivisionError):
        return False
    return True


def _looks_like_quantum_number_triplet(su3: Any, su2: Any, hypercharge: Any) -> bool:
    return (
        _is_valid_su3_value(su3)
        and _is_valid_su2_value(su2)
        and _is_valid_hypercharge_value(hypercharge)
    )


def _make_pending_confirmation_from_direct_request(question: str) -> PendingBuildConfirmation | None:
    payload = _extract_direct_field_request(question)
    if payload is None:
        return None
    return PendingBuildConfirmation(
        tool_name=payload["tool_name"],
        tool_args=_normalize_tool_args(payload["tool_args"]),
        tool_call_id="direct_build_call",
        requires_kind_confirmation=bool(payload["requires_kind_confirmation"]),
    )


def _make_pending_confirmation_from_llm_field_interpretation(
    question: str,
    *,
    llm: Any,
) -> tuple[PendingBuildConfirmation | None, str | None]:
    messages: list[Any] = [
        SystemMessage(content=SYSTEM_PROMPT),
        SystemMessage(content=FIELD_INTERPRETATION_PROMPT),
        HumanMessage(content=question),
    ]
    response = _normalize_ai_response(_invoke_llm_with_retry(messages, llm))
    pending = _pending_confirmation_from_tool_calls(
        response.tool_calls,
        requires_kind_confirmation=not _question_explicitly_mentions_field_kind(question),
        question=question,
    )
    if pending is not None:
        return pending, None
    content = str(response.content or "").strip()
    return None, content or None


def _confirmation_message(pending: PendingBuildConfirmation) -> str:
    fields = _pending_confirmation_fields(pending)
    interpreted = _format_confirmation_field_content(fields)
    if pending.requires_kind_confirmation:
        return (
            f"I interpreted the BSM field content as {interpreted}. "
            "You did not specify whether the new field is a scalar or fermion. "
            "Reply with either 'scalar' or 'fermion' if these quantum numbers are correct, "
            "or restate the full field content."
        )
    return (
        f"I interpreted the BSM field content as {interpreted}. "
        "If this is correct, type only yes. If not, restate the quantum numbers."
    )


def _summarize_confirmed_result(
    tool_name: str,
    result: dict[str, Any],
) -> str:
    if tool_name == "build_report":
        field_text = _format_field_content(result.get("fields") or []) if result.get("fields") else "SM"
        pdf_text = result["pdf_path"] or "PDF was not built because pdflatex is unavailable"
        return (
            f"Generated report for '{result['model_name']}' with field content {field_text}. "
            f"TeX: {result['tex_path']}. PDF: {pdf_text}."
        )

    anomaly_hint = _anomaly_cancellation_hint(result)
    if anomaly_hint is not None:
        return f"The model is constructed, but it is anomalous. {anomaly_hint}"
    return "The model is constructed."


def _execute_pending_confirmation(
    pending: PendingBuildConfirmation,
    *,
    history: list[Any],
    question: str,
    current_model: CurrentModelState | None,
) -> tuple[str, list[Any], PendingBuildConfirmation | None, CurrentModelState | None]:
    updated_history = [*history, HumanMessage(content=question)]
    tool_call = {
        "name": pending.tool_name,
        "args": pending.tool_args,
        "id": pending.tool_call_id,
    }
    try:
        tool_message = _tool_result_message(tool_call)
    except ValueError:
        answer = "Please restate the quantum numbers for the BSM field content."
        updated_history.append(AIMessage(content=answer))
        return answer, updated_history, None, current_model

    payload = json.loads(tool_message.content)
    answer = _summarize_confirmed_result(pending.tool_name, payload)
    updated_history.extend([tool_message, AIMessage(content=answer)])
    next_model = current_model
    if pending.tool_name == "build_model":
        next_model = CurrentModelState(result=payload)
    return answer, updated_history, None, next_model




def _is_single_edit_typo(candidate: str, target: str) -> bool:
    if candidate == target:
        return True
    if abs(len(candidate) - len(target)) > 1:
        return False
    if len(candidate) == len(target):
        mismatches = [i for i, (left, right) in enumerate(zip(candidate, target)) if left != right]
        if len(mismatches) == 1:
            return True
        if len(mismatches) == 2:
            i, j = mismatches
            return j == i + 1 and candidate[i] == target[j] and candidate[j] == target[i]
        return False
    if len(candidate) + 1 == len(target):
        short, long = candidate, target
    elif len(target) + 1 == len(candidate):
        short, long = target, candidate
    else:
        return False
    index_short = 0
    index_long = 0
    skipped = False
    while index_short < len(short) and index_long < len(long):
        if short[index_short] == long[index_long]:
            index_short += 1
            index_long += 1
            continue
        if skipped:
            return False
        skipped = True
        index_long += 1
    return True


_COMMAND_TYPO_KEYWORDS = {
    "pdf",
    "report",
    "tex",
    "latex",
    "expand",
    "expanded",
    "mass",
    "masses",
    "matrix",
    "matrices",
    "generate",
    "save",
    "file",
    "write",
    "ewsb",
}


def _normalize_command_typos(question: str) -> str:
    words = re.findall(r"[a-z0-9]+|[^a-z0-9]+", question.lower())
    normalized: list[str] = []
    for piece in words:
        if not piece.isalnum():
            normalized.append(piece)
            continue
        replacement = piece
        if len(piece) >= 3:
            matches = [keyword for keyword in _COMMAND_TYPO_KEYWORDS if _is_single_edit_typo(piece, keyword)]
            exact_matches = [keyword for keyword in matches if keyword == piece]
            if exact_matches:
                replacement = exact_matches[0]
            elif len(matches) == 1:
                replacement = matches[0]
        normalized.append(replacement)
    return "".join(normalized)

def _looks_like_current_model_summary_request(question: str) -> bool:
    lowered = " ".join(question.lower().split())
    if lowered in {
        "describe",
        "summary",
        "summarize",
        "summarise",
        "describe it",
        "summarize it",
        "summarise it",
    }:
        return True
    summary_tokens = (
        "summary",
        "summarize",
        "summarise",
        "summarize the model",
        "summarise the model",
        "summarize model",
        "summarise model",
        "summarize this model",
        "summarise this model",
        "describe the model",
        "describe model",
        "describe this model",
        "describe current model",
        "current model",
        "this model",
        "kinetic terms",
        "scalar potential",
        "yukawa",
        "anomaly",
        "anomaly free",
        "anomalous",
    )
    return any(token in lowered for token in summary_tokens)


def _looks_like_lagrangian_expansion_request(question: str) -> bool:
    lowered = " ".join(_normalize_command_typos(question).split())
    condensed = lowered.replace(" ", "")
    if lowered in {"expand", "expand it", "expand this", "expanded", "full expansion"}:
        return True
    return "expand" in lowered and any(stem in condensed for stem in ("lagrangian", "lagran", "lagrangi", "lagrang"))


def _looks_like_report_request(question: str) -> bool:
    lowered = " ".join(_normalize_command_typos(question).split())
    report_tokens = (
        "pdf",
        "report",
        "tex",
        "latex",
        "file",
        "save",
        "write out",
        "generate pdf",
        "add expanded",
        "expanded lagrangian",
        "expand the lagrangian",
        "expand lagrangian",
        "add mass matrices",
        "mass matrices",
        "mass matrix",
        "generate masses",
        "generate mass matrices",
        "generate mass matrix",
        "section 3",
        "section 4",
    )
    return any(token in lowered for token in report_tokens) or _looks_like_lagrangian_expansion_request(question)


def _wants_expanded_lagrangian(question: str) -> bool:
    lowered = " ".join(_normalize_command_typos(question).split())
    return any(
        token in lowered
        for token in (
            "add expanded",
            "expanded lagrangian",
            "expand the lagrangian",
            "expand lagrangian",
            "full lagrangian",
            "section 3",
        )
    ) or _looks_like_lagrangian_expansion_request(question)


def _wants_mass_matrices(question: str) -> bool:
    lowered = " ".join(_normalize_command_typos(question).split())
    return any(
        token in lowered
        for token in (
            "add mass matrices",
            "add mass matrix",
            "mass matrices",
            "mass matrix",
            "generate masses",
            "generate mass matrices",
            "generate mass matrix",
            "section 4",
            "ewsb",
        )
    )


def _has_neutral_component(su2_dimension: int, hypercharge: str) -> bool:
    charge_offset = Fraction(str(hypercharge))
    highest_two_m = su2_dimension - 1
    return any(Fraction(two_m, 2) + charge_offset == 0 for two_m in range(highest_two_m, -highest_two_m - 1, -2))


def _auto_ewsb_config_for_current_model(current_result: dict[str, Any]) -> dict[str, Any]:
    vevs: list[dict[str, str]] = [{"field": "H", "symbol": "v"}]
    for field in current_result.get("fields") or []:
        if field.get("kind") != "scalar":
            continue
        if str(field.get("su3")) != "1":
            continue
        try:
            su2_dimension = int(field.get("su2"))
        except (TypeError, ValueError):
            continue
        hypercharge = str(field.get("hypercharge", "")).strip()
        if not hypercharge or not _has_neutral_component(su2_dimension, hypercharge):
            continue
        name = str(field.get("name", "")).strip()
        if not name or name == "H":
            continue
        vevs.append({"field": name, "symbol": f"v_{name}"})
    return {"vevs": vevs}


def _updated_report_config(question: str, current_result: dict[str, Any]) -> dict[str, Any]:
    config = dict(current_result.get("_report_config") or {"expand_sectors": [], "ewsb": None})
    explicit_sectors = _extract_report_expand_sectors(question)
    if explicit_sectors is not None:
        config["expand_sectors"] = explicit_sectors
    elif _wants_expanded_lagrangian(question):
        config["expand_sectors"] = ["yukawa", "scalar_potential", "gauge_kinetic"]
    elif not current_result.get("_report_config"):
        config["expand_sectors"] = []

    if _wants_mass_matrices(question):
        config["ewsb"] = _auto_ewsb_config_for_current_model(current_result)
    elif not current_result.get("_report_config"):
        config["ewsb"] = None
    return config


def _extract_report_expand_sectors(question: str) -> list[str] | None:
    lowered = " ".join(_normalize_command_typos(question).split())
    explicit_only = "only expand" in lowered or "expand only" in lowered
    sectors: list[str] = []

    if any(token in lowered for token in ("yukawa", "yukawas")):
        sectors.append("yukawa")
    if any(token in lowered for token in ("scalar sector", "scalar potential", "scalar terms")):
        sectors.append("scalar_potential")
    if any(token in lowered for token in ("gauge kinetic", "gauge sector", "kinetic terms", "gauge and kinetic")):
        sectors.append("gauge_kinetic")

    if explicit_only and sectors:
        return sectors

    if any(
        token in lowered
        for token in (
            "do not expand the gauge kinetic",
            "do not expand gauge kinetic",
            "without expanding the gauge kinetic",
            "without expanding gauge kinetic",
        )
    ):
        return ["yukawa", "scalar_potential"]

    if any(
        token in lowered
        for token in (
            "do not expand the scalar",
            "without expanding the scalar",
            "keep the scalar sector compact",
        )
    ):
        return ["yukawa", "gauge_kinetic"]

    if any(
        token in lowered
        for token in (
            "do not expand the yukawa",
            "without expanding the yukawa",
            "keep the yukawa sector compact",
        )
    ):
        return ["scalar_potential", "gauge_kinetic"]

    return None


def _su3_rep_name(rep: str) -> str:
    names = {
        "1": "color singlet",
        "3": "color triplet",
        "bar3": "color anti-triplet",
        "3*": "color anti-triplet",
        "6": "color sextet",
        "bar6": "color anti-sextet",
        "6*": "color anti-sextet",
        "8": "color octet",
        "10": "color decuplet",
        "bar10": "color anti-decuplet",
    }
    return names.get(str(rep), f"SU(3) representation {rep}")


def _su2_rep_name(rep: int) -> str:
    names = {
        1: "weak singlet",
        2: "weak doublet",
        3: "weak triplet",
        4: "weak quartet",
        5: "weak quintet",
        6: "weak sextet",
        7: "weak septet",
    }
    return names.get(int(rep), f"SU(2) representation {rep}")


def _clean_display_text(text: str) -> str:
    cleaned = str(text)
    cleaned = cleaned.replace("**", "")
    cleaned = cleaned.replace("*   ", "  ")
    cleaned = cleaned.replace("* ", "- ")
    cleaned = cleaned.replace("$_L$", "")
    cleaned = cleaned.replace("SU(2)L", "SU(2)")
    cleaned = cleaned.replace("SU(2)_L", "SU(2)")
    cleaned = cleaned.replace("$", "")
    return cleaned


def _format_field_description(field: dict[str, Any]) -> str:
    name = str(field.get("name") or "Field")
    kind = str(field.get("kind") or "field")
    su3 = str(field.get("su3"))
    su2 = int(field.get("su2"))
    hypercharge = str(field.get("hypercharge"))
    return (
        f"{name} is a {kind} in {_field_signature(field)}: "
        f"{_su3_rep_name(su3)}, {_su2_rep_name(su2)}, hypercharge {hypercharge}."
    )


def _format_lagrangian_summary_lines(summary: str) -> list[str]:
    cleaned = _clean_display_text(summary).strip()
    if not cleaned:
        return []
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if lines and lines[0].rstrip(":").lower() == "lagrangian summary":
        lines = lines[1:]

    formatted: list[str] = []
    for line in lines:
        if not line:
            formatted.append("  -")
            continue
        label, separator, value = line.partition(":")
        label_text = label.replace("_", " ")
        label_text = f"{label_text[0].upper()}{label_text[1:]}" if label_text else label_text
        formatted.append(f"  - {label_text}{separator}{value}")
    return formatted


def _format_signed_fraction(value: Any) -> str:
    fraction = Fraction(str(value))
    return str(fraction) if fraction < 0 else f"+{fraction}"


def _anomaly_cancellation_hint(result: dict[str, Any]) -> str | None:
    anomaly_summary = _clean_display_text(str(result.get("anomaly_summary") or "")).strip()
    if not anomaly_summary or anomaly_summary.startswith("anomaly-free:"):
        return None

    fields = [field for field in result.get("fields") or [] if isinstance(field, dict)]
    fermions = [field for field in fields if str(field.get("kind") or "").strip().lower() in {"fermion", "weyl_fermion"}]
    if len(fermions) != 1 or len(fields) != 1:
        return None

    field = fermions[0]
    try:
        conjugate_su3 = SU3Rep.parse(str(field.get("su3"))).conjugate.label()
        su2 = int(field.get("su2"))
        hypercharge = _format_signed_fraction(-Fraction(str(field.get("hypercharge"))))
    except (TypeError, ValueError, ZeroDivisionError):
        return None

    return f"Try adding the conjugate fermion ({conjugate_su3},{su2},{hypercharge}) to cancel the anomaly."


def _format_current_model_summary(current_model: CurrentModelState) -> str:
    result = current_model.result
    fields = list(result.get("fields") or [])
    mixed_bsm_count = len(result.get("mixed_bsm_terms") or [])
    anomaly_summary = _clean_display_text(str(result.get("anomaly_summary") or ""))
    anomaly_hint = _anomaly_cancellation_hint(result)
    lagrangian_summary_lines = _format_lagrangian_summary_lines(str(result.get("lagrangian_summary") or ""))
    is_anomaly_free = anomaly_summary.startswith("anomaly-free:")
    anomaly_status_text = _highlight_anomaly_free("anomaly-free") if is_anomaly_free else _highlight_anomalous("anomalous")
    lines = [f"Model: {result.get('model_name', 'BSM model')}"]
    if fields:
        lines.append("Field content:")
        lines.extend(f"  - {_format_field_description(field)}" for field in fields)
    lines.extend(["", "Interactions and consistency:"])
    lines.extend(lagrangian_summary_lines)
    lines.append(f"  - Mixed BSM interaction terms: {mixed_bsm_count}")
    lines.append(f"  - Status: {anomaly_status_text}")
    lines.append(f"  - Anomaly summary: {anomaly_summary}")
    if anomaly_hint is not None:
        lines.append(f"  - Hint: {anomaly_hint}")
    return "\n".join(lines)


def _default_model_name_from_fields(fields: list[dict[str, Any]]) -> str:
    if not fields:
        return "SM"
    return "SM+" + "+".join(str(field.get("name") or "").strip() for field in fields)


def _model_name_for_updated_fields(current_model: CurrentModelState, updated_fields: list[dict[str, Any]]) -> str:
    current_name = str(current_model.result.get("model_name") or "").strip()
    old_fields = list(current_model.result.get("fields") or [])
    old_default = _default_model_name_from_fields(old_fields)
    if current_name == old_default or not current_name:
        return _default_model_name_from_fields(updated_fields)
    return current_name


def _normalize_field_reference(text: str) -> str:
    normalized = str(text).strip().strip("`").rstrip(".,;:!?")
    normalized = normalized.replace(" ", "")
    normalized = normalized.replace("{", "").replace("}", "")
    normalized = normalized.replace("\\", "")
    return normalized.lower()


def _extract_field_rename_request(question: str) -> tuple[str, str] | None:
    match = re.search(
        r"(?:rename|change)\s+([\\A-Za-z0-9_^{}]+)\s+(?:to|into|as)\s+([\\A-Za-z0-9_^{}]+)",
        question,
        re.IGNORECASE,
    )
    if match is None:
        return None
    old_name = match.group(1).strip().rstrip(".,;:!?")
    new_name = match.group(2).strip().rstrip(".,;:!?")
    if not old_name or not new_name:
        return None
    return old_name, new_name


def _apply_report_field_aliases(question: str, fields: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rename = _extract_field_rename_request(question)
    if rename is None:
        return [dict(field) for field in fields]
    old_name, new_name = rename
    normalized_old = _normalize_field_reference(old_name)
    updated_fields = [dict(field) for field in fields]
    for field in updated_fields:
        current_name = str(field.get("name") or "").strip()
        variants = {
            _normalize_field_reference(current_name),
            _normalize_field_reference(field_latex_name(current_name)),
        }
        if normalized_old in variants:
            field["latex_name"] = new_name
            break
    return updated_fields


def _rename_current_model_field(
    question: str,
    *,
    current_model: CurrentModelState,
) -> tuple[str, CurrentModelState] | None:
    rename = _extract_field_rename_request(question)
    if rename is None:
        return None
    old_name, new_name = rename
    normalized_old = _normalize_field_reference(old_name)
    fields = list(current_model.result.get("fields") or [])
    target_index: int | None = None
    for index, field in enumerate(fields):
        current_name = str(field.get("name") or "").strip()
        variants = {
            _normalize_field_reference(current_name),
            _normalize_field_reference(field_latex_name(current_name)),
        }
        if normalized_old in variants:
            target_index = index
            break
    if target_index is None:
        return None

    if any(
        idx != target_index and str(field.get("name") or "").strip() == new_name
        for idx, field in enumerate(fields)
    ):
        answer = f"The current model already has a field named {new_name}."
        return answer, current_model

    old_default = _default_model_name_from_fields(fields)
    updated_fields = [dict(field) for field in fields]
    previous_name = str(updated_fields[target_index].get("name") or "").strip()
    updated_fields[target_index]["name"] = new_name

    updated_result = dict(current_model.result)
    updated_result["fields"] = updated_fields
    current_name = str(current_model.result.get("model_name") or "").strip()
    if current_name == old_default:
        updated_result["model_name"] = _default_model_name_from_fields(updated_fields)

    answer = f"Renamed {previous_name} to {new_name} in the current model."
    return answer, CurrentModelState(result=updated_result)


def _looks_like_addition_request(question: str) -> bool:
    lowered = " ".join(question.lower().split())
    return any(
        token in lowered
        for token in (
            "add ",
            "append ",
            "include ",
            "introduce ",
            "insert ",
            "augment ",
            "also add ",
            "also include ",
            "also introduce ",
            "plus ",
            "another ",
            "an additional ",
            "additional ",
            "increase the field content by ",
            "extend the field content by ",
            "enlarge the field content by ",
            "with an additional ",
            "with additional ",
        )
    )


def _looks_like_clear_current_model_request(question: str) -> bool:
    lowered = " ".join(question.lower().split())
    exact_commands = {
        "clear",
        "remove",
        "reset",
        "clear model",
        "clear the model",
        "clear current model",
        "clear the current model",
        "remove model",
        "remove the model",
        "remove current model",
        "remove the current model",
        "delete model",
        "delete the model",
        "reset model",
        "reset the model",
        "reset current model",
        "reset the current model",
        "start over",
    }
    return lowered in exact_commands


def _looks_like_new_model_request(question: str) -> bool:
    lowered = " ".join(question.lower().split())
    return any(
        phrase in lowered
        for phrase in (
            "new model",
            "another model",
            "fresh model",
            "separate model",
            "from scratch",
            "start a new model",
            "build me a new model",
            "make me a new model",
        )
    )


def _looks_like_current_model_field_request(question: str) -> bool:
    lowered = " ".join(question.lower().split())
    if _looks_like_addition_request(question):
        return True
    field_cues = (
        "neutrino",
        "right handed",
        "left handed",
        "sterile",
        "scalar",
        "fermion",
        "singlet",
        "doublet",
        "triplet",
        "quartet",
        "quintet",
        "sextet",
        "septet",
        "octet",
        "decuplet",
        "color",
        "colour",
        "weak",
        "hypercharge",
        "su2",
        "su3",
        "generation",
        "generations",
        "field content",
        "new field",
        "extra field",
        "additional field",
        "additional state",
        "new state",
        "extra state",
    )
    return any(cue in lowered for cue in field_cues)


def _merge_fields_into_current_model(
    current_model: CurrentModelState,
    new_fields: list[dict[str, Any]],
) -> dict[str, Any]:
    existing_fields = [dict(field) for field in current_model.result.get("fields") or []]
    merged_fields = [*existing_fields, *new_fields]
    payload = {"fields": merged_fields}
    current_name = str(current_model.result.get("model_name") or "").strip()
    old_default = _default_model_name_from_fields(existing_fields)
    merged_default = _default_model_name_from_fields(merged_fields)
    all_named = all(str(field.get("name") or "").strip() for field in merged_fields)
    if current_name and current_name != old_default:
        payload["model_name"] = current_name
    elif all_named:
        payload["model_name"] = merged_default
    return payload


def _make_pending_confirmation_for_current_model_addition(
    question: str,
    *,
    llm: Any,
    current_model: CurrentModelState,
) -> tuple[PendingBuildConfirmation | None, str | None]:
    try:
        direct_pending = _make_pending_confirmation_from_direct_request(question)
    except Exception as exc:
        return _fallback_to_llm_clarification(
            question,
            llm=llm,
            history=history,
            current_model=current_model,
            detail=str(exc),
        )
    if direct_pending is not None:
        new_fields = list(direct_pending.tool_args.get("fields") or [])
        merged_args = _merge_fields_into_current_model(current_model, new_fields)
        return (
            PendingBuildConfirmation(
                tool_name=direct_pending.tool_name,
                tool_args=merged_args,
                tool_call_id=direct_pending.tool_call_id,
                requires_kind_confirmation=direct_pending.requires_kind_confirmation,
                confirmation_fields=[field for field in new_fields if isinstance(field, dict)],
            ),
            None,
        )

    messages: list[Any] = [
        SystemMessage(content=SYSTEM_PROMPT),
        SystemMessage(content=CURRENT_MODEL_ADDITION_PROMPT),
        HumanMessage(content=question),
    ]
    response = _normalize_ai_response(_invoke_llm_with_retry(messages, llm))
    pending = _pending_confirmation_from_tool_calls(
        response.tool_calls,
        requires_kind_confirmation=not _question_explicitly_mentions_field_kind(question),
        question=question,
    )
    if pending is not None:
        new_fields = list(pending.tool_args.get("fields") or [])
        merged_args = _merge_fields_into_current_model(current_model, new_fields)
        return (
            PendingBuildConfirmation(
                tool_name=pending.tool_name,
                tool_args=merged_args,
                tool_call_id=pending.tool_call_id,
                requires_kind_confirmation=pending.requires_kind_confirmation,
                confirmation_fields=[field for field in new_fields if isinstance(field, dict)],
            ),
            None,
        )
    content = str(response.content or "").strip()
    return None, content or None


def _looks_like_sm_only_report_request(question: str) -> bool:
    lowered = " ".join(question.lower().split())
    requests_report = any(token in lowered for token in ("pdf", "report", "tex", "latex"))
    tokens = re.sub(r"[^a-z0-9+]+", " ", lowered).split()
    mentions_sm = "sm" in tokens or "standard model" in lowered
    excludes_bsm = not any(token in lowered for token in ("bsm", "extend", "extension", "new field", "extra field"))
    return requests_report and mentions_sm and excludes_bsm


def _handle_direct_request(question: str) -> tuple[str, CurrentModelState | None] | None:
    if _looks_like_sm_only_report_request(question):
        result = build_report_from_payload(
            {
                "fields": [],
                "model_name": "SM",
                "output_stem": "SM_report",
                "output_dir": "output",
                "make_pdf": True,
                "expand_sectors": [],
            },
            base_dir=Path.cwd(),
        )
        pdf_text = str(result.pdf_path) if result.pdf_path else "PDF was not built because pdflatex is unavailable"
        answer = (
            f"Generated the Standard Model report. "
            f"TeX: {result.tex_path}. PDF: {pdf_text}."
        )
        current_result = {
            "model_name": "SM",
            "fields": [],
            "_report_config": {"expand_sectors": [], "ewsb": None},
            "_report_output_stem": "SM_report",
        }
        return answer, CurrentModelState(result=current_result)
    return None


def _handle_current_model_report_request(
    question: str,
    *,
    current_model: CurrentModelState,
) -> tuple[str, CurrentModelState] | None:
    if not _looks_like_report_request(question):
        return None

    result = current_model.result
    lowered = " ".join(question.lower().split())
    make_pdf = "tex only" not in lowered and "only tex" not in lowered
    fields = [dict(field) for field in result.get("fields") or []]
    fields = _apply_report_field_aliases(question, fields)
    report_config = _updated_report_config(question, result)
    output_stem = str(
        result.get("_report_output_stem")
        or str(result.get("model_name") or "bsm_report").replace("+", "_")
    )
    payload = {
        "fields": fields,
        "model_name": result.get("model_name"),
        "output_stem": output_stem,
        "output_dir": "output",
        "make_pdf": make_pdf,
        "expand_sectors": report_config.get("expand_sectors") or [],
    }
    if report_config.get("ewsb") is not None:
        payload["ewsb"] = report_config["ewsb"]

    report = build_report_from_payload(payload, base_dir=Path.cwd())
    pdf_text = str(report.pdf_path) if report.pdf_path else "PDF was not built because pdflatex is unavailable"
    updated_result = dict(result)
    updated_result["_report_config"] = report_config
    updated_result["_report_output_stem"] = report.output_stem
    answer = (
        f"Generated report for '{report.model_name}' with field content "
        f"{_format_field_content(result.get('fields') or []) if result.get('fields') else 'SM'}. "
        f"TeX: {report.tex_path}. PDF: {pdf_text}."
    )
    return answer, CurrentModelState(result=updated_result)


def _fallback_to_llm_clarification(
    question: str,
    *,
    llm: Any,
    history: list[Any],
    current_model: CurrentModelState | None,
    detail: str | None = None,
) -> tuple[str, list[Any], PendingBuildConfirmation | None, CurrentModelState | None]:
    context_note = ""
    if current_model is not None:
        model_name = str(current_model.result.get("model_name") or "current model")
        context_note = f"Current model: {model_name}. "
    detail_note = f" Internal detail: {detail}." if detail else ""
    messages: list[Any] = [
        SystemMessage(content=SYSTEM_PROMPT),
        *history,
        HumanMessage(content=question),
        HumanMessage(
            content=(
                context_note
                + "The structured parsing or execution path could not safely complete this request. "
                "Do not call any tool. Ask the user a short clarification question or restate your interpretation and ask whether it is correct before proceeding."
                + detail_note
            )
        ),
    ]
    response = _normalize_ai_response(_invoke_llm_with_retry(messages, llm))
    updated_history = [*history, HumanMessage(content=question), response]
    return str(response.content or ""), updated_history, None, current_model


def _invoke_llm_with_retry(messages: list[Any], llm: Any) -> Any:
    try:
        return llm.invoke(messages)
    except Exception as exc:
        detail = str(exc)
        if "error parsing tool call" not in detail.lower():
            raise

        retry_messages = [
            *messages,
            HumanMessage(
                content=(
                    "Retry the previous answer. If you call a tool, the arguments must be strict JSON. "
                    "Use only double-quoted strings. "
                    "Do not emit fractions like 1/6 directly; emit them as strings like \"1/6\". "
                    "For a pure Standard Model report, use fields as an empty list and model_name as \"SM\"."
                )
            ),
        ]
        try:
            return llm.invoke(retry_messages)
        except Exception as retry_exc:
            retry_detail = str(retry_exc)
            if "error parsing tool call" not in retry_detail.lower():
                raise
            plain_llm = llm.bind_tools([]) if hasattr(llm, "bind_tools") else llm
            plain_messages = [
                *messages,
                HumanMessage(
                    content=(
                        "Do not call any tool. Answer in plain text only. "
                        "If the request is ambiguous, briefly restate your interpretation and ask the user to confirm it."
                    )
                ),
            ]
            return plain_llm.invoke(plain_messages)


def ask(
    question: str,
    *,
    llm: Any,
    history: list[Any] | None = None,
    current_model: CurrentModelState | None = None,
    pending_confirmation: PendingBuildConfirmation | None = None,
    max_round_trips: int = 8,
) -> tuple[str, list[Any], PendingBuildConfirmation | None, CurrentModelState | None]:
    if history is None:
        history = []

    if pending_confirmation is not None:
        normalized = question.strip().lower()
        requested_kind = _extract_explicit_field_kind(question)
        if pending_confirmation.requires_kind_confirmation:
            if requested_kind is not None:
                updated_pending = PendingBuildConfirmation(
                    tool_name=pending_confirmation.tool_name,
                    tool_args=_apply_field_kind(pending_confirmation.tool_args, requested_kind),
                    tool_call_id=pending_confirmation.tool_call_id,
                    requires_kind_confirmation=False,
                )
                try:
                    return _execute_pending_confirmation(
                        updated_pending,
                        history=history,
                        question=question,
                        current_model=current_model,
                    )
                except Exception as exc:
                    return _fallback_to_llm_clarification(
                        question,
                        llm=llm,
                        history=history,
                        current_model=current_model,
                        detail=str(exc),
                    )
            if normalized == "yes":
                answer = "Please specify whether the new field is a scalar or fermion."
                updated_history = [*history, HumanMessage(content=question), AIMessage(content=answer)]
                return answer, updated_history, pending_confirmation, current_model
        if normalized == "yes":
            try:
                return _execute_pending_confirmation(
                    pending_confirmation,
                    history=history,
                    question=question,
                    current_model=current_model,
                )
            except Exception as exc:
                return _fallback_to_llm_clarification(
                    question,
                    llm=llm,
                    history=history,
                    current_model=current_model,
                    detail=str(exc),
                )
        if normalized in {"no", "n"}:
            answer = "Please restate the quantum numbers for the BSM field content."
            updated_history = [*history, HumanMessage(content=question), AIMessage(content=answer)]
            return answer, updated_history, None, current_model

        answer = (
            "Please confirm the pending field content first by replying 'yes', "
            "or reply 'no' and restate the quantum numbers."
        )
        updated_history = [*history, HumanMessage(content=question), AIMessage(content=answer)]
        return answer, updated_history, pending_confirmation, current_model

    if current_model is not None and _looks_like_clear_current_model_request(question):
        answer = "Cleared the current model."
        updated_history = [*history, HumanMessage(content=question), AIMessage(content=answer)]
        return answer, updated_history, None, None

    if (
        current_model is not None
        and _looks_like_current_model_summary_request(question)
        and not _looks_like_report_request(question)
    ):
        answer = _format_current_model_summary(current_model)
        updated_history = [*history, HumanMessage(content=question), AIMessage(content=answer)]
        return answer, updated_history, None, current_model

    fresh_model_request = current_model is not None and _looks_like_new_model_request(question)
    if fresh_model_request and not (
        _looks_like_field_content_statement(question) or _looks_like_multi_field_request(question)
    ):
        answer = (
            "What fields would you like to include in the new model? "
            "Please list each field's SU(3), SU(2), hypercharge, and whether it is a scalar or fermion."
        )
        updated_history = [*history, HumanMessage(content=question), AIMessage(content=answer)]
        return answer, updated_history, None, None

    if current_model is not None and not fresh_model_request:
        try:
            report_result = _handle_current_model_report_request(question, current_model=current_model)
        except Exception as exc:
            return _fallback_to_llm_clarification(
                question,
                llm=llm,
                history=history,
                current_model=current_model,
                detail=str(exc),
            )
        if report_result is not None:
            report_answer, next_model = report_result
            updated_history = [*history, HumanMessage(content=question), AIMessage(content=report_answer)]
            return report_answer, updated_history, None, next_model

        rename_result = _rename_current_model_field(question, current_model=current_model)
        if rename_result is not None:
            answer, next_model = rename_result
            updated_history = [*history, HumanMessage(content=question), AIMessage(content=answer)]
            return answer, updated_history, None, next_model

        if _looks_like_current_model_field_request(question):
            try:
                add_pending, add_answer = _make_pending_confirmation_for_current_model_addition(
                    question,
                    llm=llm,
                    current_model=current_model,
                )
            except Exception as exc:
                return _fallback_to_llm_clarification(
                    question,
                    llm=llm,
                    history=history,
                    current_model=current_model,
                    detail=str(exc),
                )
            if add_pending is not None:
                answer = _confirmation_message(add_pending)
                updated_history = [*history, HumanMessage(content=question), AIMessage(content=answer)]
                return answer, updated_history, add_pending, current_model
            if add_answer is not None:
                updated_history = [*history, HumanMessage(content=question), AIMessage(content=add_answer)]
                return add_answer, updated_history, None, current_model

    try:
        direct_answer = _handle_direct_request(question)
    except Exception as exc:
        return _fallback_to_llm_clarification(
            question,
            llm=llm,
            history=history,
            current_model=current_model,
            detail=str(exc),
        )
    if direct_answer is not None:
        direct_text, next_model = direct_answer
        updated_history = [*history, HumanMessage(content=question), AIMessage(content=direct_text)]
        return direct_text, updated_history, None, next_model

    direct_pending = _make_pending_confirmation_from_direct_request(question)
    if direct_pending is not None:
        answer = _confirmation_message(direct_pending)
        updated_history = [*history, HumanMessage(content=question), AIMessage(content=answer)]
        return answer, updated_history, direct_pending, current_model

    if _looks_like_field_content_candidate(question):
        llm_pending, llm_answer = _make_pending_confirmation_from_llm_field_interpretation(question, llm=llm)
        if llm_pending is not None:
            answer = _confirmation_message(llm_pending)
            updated_history = [*history, HumanMessage(content=question), AIMessage(content=answer)]
            return answer, updated_history, llm_pending, current_model
        if llm_answer is not None:
            updated_history = [*history, HumanMessage(content=question), AIMessage(content=llm_answer)]
            return llm_answer, updated_history, None, current_model

    messages: list[Any] = [SystemMessage(content=SYSTEM_PROMPT), *history, HumanMessage(content=question)]
    updated_history = [*history, HumanMessage(content=question)]

    for _ in range(max_round_trips):
        response = _normalize_ai_response(_invoke_llm_with_retry(messages, llm))

        if not response.tool_calls:
            messages.append(response)
            updated_history.append(response)
            return response.content, updated_history, None, current_model

        pending = _pending_confirmation_from_tool_calls(
            response.tool_calls,
            requires_kind_confirmation=not _question_explicitly_mentions_field_kind(question),
            question=question,
        )
        if pending is not None:
            answer = _confirmation_message(pending)
            updated_history.append(AIMessage(content=answer))
            return answer, updated_history, pending, current_model

        messages.append(response)
        updated_history.append(response)

        try:
            tool_messages = [_tool_result_message(tool_call) for tool_call in response.tool_calls]
        except (KeyError, ValueError, TypeError, json.JSONDecodeError):
            fallback_messages = [
                *messages,
                HumanMessage(
                    content=(
                        "Your previous structured tool call could not be executed. "
                        "Do not call any tool. Answer the user's last message directly in plain text."
                    )
                ),
            ]
            fallback_response = _normalize_ai_response(_invoke_llm_with_retry(fallback_messages, llm))
            updated_history.append(fallback_response)
            return str(fallback_response.content or ""), updated_history, None, current_model

        messages.extend(tool_messages)
        updated_history.extend(tool_messages)

    raise RuntimeError("Agent exceeded the maximum number of tool round trips.")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BSM model-building chat agent")
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Model name or provider selector, e.g. qwen3.5:35b, gpt-oss:20b, "
            "openai:gpt-4.1, anthropic:claude-sonnet-4-20250514"
        ),
    )
    parser.add_argument("--api-base", default=None, help="Remote API base URL")
    parser.add_argument("--api-key", default=None, help="Remote API key/token")
    parser.add_argument("--api-email", default=None, help="Remote API email for Open WebUI sign-in")
    parser.add_argument("--api-password", default=None, help="Remote API password for Open WebUI sign-in")
    parser.add_argument(
        "--prompt",
        default=None,
        help="Single prompt to run non-interactively. If omitted, start a REPL.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    target = resolve_model_target(
        args.model,
        api_base=args.api_base,
        api_key=args.api_key,
        api_email=args.api_email,
        api_password=args.api_password,
    )

    if target.provider == "ollama":
        print(f"[INFO] Loading local model '{target.model}' via Ollama...")
    else:
        print(f"[INFO] Loading remote model '{target.model}' via {target.api_base}...")

    llm = build_chat_model(target, tools=TOOLS)
    print("[INFO] BSM agent ready.")

    history: list[Any] = []
    current_model: CurrentModelState | None = None
    pending_confirmation: PendingBuildConfirmation | None = None

    if args.prompt:
        started_at = time.perf_counter()
        with _activity_indicator():
            answer, _, pending_confirmation, current_model = ask(
                args.prompt,
                llm=llm,
                history=history,
                current_model=current_model,
            )
        elapsed_seconds = time.perf_counter() - started_at
        print(_format_agent_output(_append_step_timing(answer, elapsed_seconds)))
        if pending_confirmation is not None:
            return 2
        return 0

    while True:
        try:
            question = input(USER_PROMPT).strip()
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            print()
            return 130

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            return 0

        try:
            started_at = time.perf_counter()
            with _activity_indicator():
                answer, history, pending_confirmation, current_model = ask(
                    question,
                    llm=llm,
                    history=history,
                    current_model=current_model,
                    pending_confirmation=pending_confirmation,
                )
            elapsed_seconds = time.perf_counter() - started_at
        except KeyboardInterrupt:
            print()
            continue
        print(_format_agent_output(_append_step_timing(answer, elapsed_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())
