"""Deterministic backend helpers for the local BSM chat agent."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
import json
import re
import shutil
import subprocess
import sys

import sympy as sp

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = Path(__file__).resolve().parent.name

from . import (
    Field,
    FieldKind,
    StandardModel,
    component_label,
    compute_mass_matrices,
    gauge_self_interactions_latex,
    neutral_scalar_vev_substitutions,
    neutral_scalar_vev_shifts,
)
from .expansion import expand_operator
from .fields import field_latex_aliases, field_latex_name
from .indexed import indexed_operator_latex
from .operators import latex_identifier
from .latex_display import render_display_equations


@dataclass(frozen=True)
class FieldSpec:
    name: str
    kind: str
    su3: str
    su2: int
    hypercharge: str
    generations: int = 1
    real: bool = False
    latex_name: str | None = None

    def to_field(self) -> Field:
        if self.kind == "scalar":
            return Field.scalar(
                self.name,
                su3=self.su3,
                su2=self.su2,
                hypercharge=Fraction(self.hypercharge),
                generations=self.generations,
                real=self.real,
            )
        if self.kind in {"fermion", "weyl_fermion"}:
            return Field.fermion(
                self.name,
                su3=self.su3,
                su2=self.su2,
                hypercharge=Fraction(self.hypercharge),
                generations=self.generations,
            )
        raise ValueError(f"Unsupported field kind: {self.kind!r}")


@dataclass(frozen=True)
class ModelBuildResult:
    model_name: str
    fields: tuple[FieldSpec, ...]
    lagrangian_summary: str
    anomaly_summary: str
    kinetic_term_count: int
    yukawa_terms: tuple[str, ...]
    fermion_mass_terms: tuple[str, ...]
    scalar_potential_terms: tuple[str, ...]
    mixed_bsm_terms: tuple[str, ...]


@dataclass(frozen=True)
class ReportResult:
    model_name: str
    output_stem: str
    output_dir: Path
    tex_path: Path
    pdf_path: Path | None
    lagrangian_summary: str
    anomaly_summary: str
    mixed_bsm_terms: tuple[str, ...]
    fields: tuple[FieldSpec, ...]


REPORT_EXPANSION_SECTORS = ("yukawa", "scalar_potential", "gauge_kinetic")


def parse_field_specs(payload: dict) -> list[FieldSpec]:
    items = payload.get("fields")
    if items is None:
        items = []
    if not isinstance(items, list):
        raise ValueError("Expected `fields` to be a list")
    specs: list[FieldSpec] = []
    seen: set[str] = set()
    scalar_count = 0
    fermion_count = 0
    for raw in items:
        if not isinstance(raw, dict):
            raise ValueError("Each field entry must be an object")
        kind = str(raw.get("kind", "")).strip().lower()
        if kind == "scalar":
            scalar_count += 1
            name = f"phi_{scalar_count}"
        elif kind in {"fermion", "weyl_fermion"}:
            fermion_count += 1
            name = f"psi_{fermion_count}"
        else:
            raise ValueError("Each field must define `kind` as scalar or fermion")
        if name in seen:
            raise ValueError(f"Duplicate field name: {name}")
        seen.add(name)
        su3 = str(raw.get("su3", "")).strip()
        su2 = int(raw.get("su2"))
        hypercharge = _normalize_fraction_string(raw.get("hypercharge"))
        generations = int(raw.get("generations", 1))
        real = bool(raw.get("real", False))
        raw_latex_name = raw.get("latex_name")
        latex_name = None if raw_latex_name is None else str(raw_latex_name).strip() or None
        if kind != "scalar":
            real = False
        specs.append(
            FieldSpec(
                name=name,
                kind=kind,
                su3=su3,
                su2=su2,
                hypercharge=hypercharge,
                generations=generations,
                real=real,
                latex_name=latex_name,
            )
        )
    return specs


def build_model_from_payload(payload: dict) -> ModelBuildResult:
    specs = tuple(parse_field_specs(payload))
    model_name = str(payload.get("model_name") or _default_model_name(specs)).strip()
    model = StandardModel() if not specs else StandardModel().extend([spec.to_field() for spec in specs], name=model_name)
    if not specs and model_name:
        model = StandardModel() if model_name == "SM" else type(model)(model_name, model.fields, model.backend, model.gauge_kinetic_terms)
    lagrangian = model.generate_lagrangian()
    return ModelBuildResult(
        model_name=model_name,
        fields=specs,
        lagrangian_summary=lagrangian.summary(),
        anomaly_summary=model.anomaly_report().summary(),
        kinetic_term_count=len(lagrangian.kinetic_terms),
        yukawa_terms=tuple(operator.text() for operator in lagrangian.by_category("yukawa")),
        fermion_mass_terms=tuple(operator.text() for operator in lagrangian.by_category("fermion_mass")),
        scalar_potential_terms=tuple(operator.text() for operator in lagrangian.by_category("scalar_potential")),
        mixed_bsm_terms=tuple(_mixed_bsm_operator_texts(lagrangian, specs)),
    )


def build_report_from_payload(payload: dict, base_dir: Path) -> ReportResult:
    base_dir = base_dir.resolve()
    build = build_model_from_payload(payload)
    specs = build.fields
    model_name = build.model_name
    output_stem = _safe_output_stem(str(payload.get("output_stem") or model_name.replace("+", "_plus_")))
    output_dir = _safe_output_dir(base_dir, payload.get("output_dir"))
    output_dir.mkdir(parents=True, exist_ok=True)
    expand_sectors = _normalize_expand_sectors(payload.get("expand_sectors"))

    model = StandardModel() if not specs else StandardModel().extend([spec.to_field() for spec in specs], name=model_name)
    if not specs and model_name:
        model = StandardModel() if model_name == "SM" else type(model)(model_name, model.fields, model.backend, model.gauge_kinetic_terms)
    lagrangian = model.generate_lagrangian()
    tex = _render_report(
        model_name,
        specs,
        model,
        lagrangian,
        expand_sectors=expand_sectors,
        ewsb_config=_normalize_ewsb_config(payload.get("ewsb")),
    )

    tex_path = output_dir / f"{output_stem}.tex"
    tex_path.write_text(tex)
    pdf_path = _write_pdf(tex_path) if payload.get("make_pdf", True) else None

    mixed_terms = tuple(_mixed_bsm_operator_texts(lagrangian, specs))
    metadata = {
        "model_name": model_name,
        "output_stem": output_stem,
        "fields": [spec.__dict__ for spec in specs],
        "lagrangian_summary": build.lagrangian_summary,
        "anomaly_summary": build.anomaly_summary,
        "mixed_bsm_terms": list(build.mixed_bsm_terms),
        "tex_path": str(tex_path),
        "pdf_path": str(pdf_path) if pdf_path else None,
        "expand_sectors": list(expand_sectors),
        "ewsb": payload.get("ewsb"),
    }
    (output_dir / f"{output_stem}.json").write_text(json.dumps(metadata, indent=2))

    return ReportResult(
        model_name=model_name,
        output_stem=output_stem,
        output_dir=output_dir,
        tex_path=tex_path,
        pdf_path=pdf_path,
        lagrangian_summary=build.lagrangian_summary,
        anomaly_summary=build.anomaly_summary,
        mixed_bsm_terms=build.mixed_bsm_terms,
        fields=specs,
    )


def _normalize_fraction_string(value: object) -> str:
    if value is None:
        raise ValueError("Each field must define `hypercharge`")
    return str(Fraction(str(value)))


def _default_model_name(specs: tuple[FieldSpec, ...]) -> str:
    return "SM+" + "+".join(spec.name for spec in specs)


def _latex_alias_map(specs: tuple[FieldSpec, ...]) -> dict[str, str]:
    return {
        spec.name: spec.latex_name
        for spec in specs
        if spec.latex_name is not None and spec.latex_name.strip()
    }


def _report_field_latex_alias_map(specs: tuple[FieldSpec, ...]) -> dict[str, str]:
    aliases = _latex_alias_map(specs)
    report_aliases: dict[str, str] = {}
    for spec in specs:
        base_name = aliases.get(spec.name, field_latex_name(spec.name))
        color = "blue" if spec.kind == "scalar" else "red"
        report_aliases[spec.name] = rf"\textcolor{{{color}}}{{{base_name}}}"
    return report_aliases


def _safe_output_stem(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in value)
    return cleaned.strip("._") or "bsm_report"


def _safe_output_dir(base_dir: Path, value: object) -> Path:
    raw = str(value or "output").strip()
    if not raw:
        raw = "output"

    candidate = Path(raw)
    if candidate.is_absolute():
        candidate = Path("output")

    normalized_parts = [part for part in candidate.parts if part not in {"", ".", ".."}]
    if not normalized_parts:
        normalized_parts = ["output"]

    resolved = (base_dir / Path(*normalized_parts)).resolve()
    try:
        resolved.relative_to(base_dir)
    except ValueError:
        return (base_dir / "output").resolve()
    return resolved


def _normalize_expand_sectors(value: object) -> tuple[str, ...]:
    if value is None:
        return ()

    if isinstance(value, str):
        raw_items = [value]
    elif isinstance(value, (list, tuple, set)):
        raw_items = [str(item) for item in value]
    else:
        raise ValueError("`expand_sectors` must be a string or a list of strings")

    alias_map = {
        "all": REPORT_EXPANSION_SECTORS,
        "full": REPORT_EXPANSION_SECTORS,
        "yukawa": ("yukawa",),
        "yukawas": ("yukawa",),
        "yukawa terms": ("yukawa",),
        "fermion mass": ("yukawa",),
        "fermion masses": ("yukawa",),
        "yukawa and fermion-mass": ("yukawa",),
        "scalar": ("scalar_potential",),
        "scalar sector": ("scalar_potential",),
        "scalar potential": ("scalar_potential",),
        "potential": ("scalar_potential",),
        "gauge": ("gauge_kinetic",),
        "kinetic": ("gauge_kinetic",),
        "gauge kinetic": ("gauge_kinetic",),
        "gauge and kinetic": ("gauge_kinetic",),
        "gauge sector": ("gauge_kinetic",),
    }

    normalized: list[str] = []
    for item in raw_items:
        lowered = " ".join(item.strip().lower().replace("_", " ").replace("-", " ").split())
        if not lowered:
            continue
        resolved = alias_map.get(lowered)
        if resolved is None and lowered in REPORT_EXPANSION_SECTORS:
            resolved = (lowered,)
        if resolved is None:
            raise ValueError(
                "Unsupported expand_sectors entry "
                f"{item!r}; expected combinations of {', '.join(REPORT_EXPANSION_SECTORS)}."
            )
        for sector in resolved:
            if sector not in normalized:
                normalized.append(sector)

    return tuple(normalized)


def _normalize_ewsb_config(value: object) -> dict | None:
    if value is None or value is False:
        return None
    if value is True:
        return {"vevs": ({"field": "H", "symbol": "v"},)}
    if not isinstance(value, dict):
        raise ValueError("`ewsb` must be a boolean or an object")
    enabled = value.get("enabled", True)
    if not enabled:
        return None

    vevs: list[dict[str, str]] = []
    if "sm_higgs_vev" in value or "vevs" not in value:
        vev_symbol = str(value.get("sm_higgs_vev", "v")).strip() or "v"
        vevs.append({"field": "H", "symbol": vev_symbol})

    raw_vevs = value.get("vevs", ())
    if raw_vevs is None:
        raw_vevs = ()
    if not isinstance(raw_vevs, list | tuple):
        raise ValueError("`ewsb.vevs` must be a list of objects")

    seen_fields = {entry["field"] for entry in vevs}
    for raw in raw_vevs:
        if not isinstance(raw, dict):
            raise ValueError("Each `ewsb.vevs` entry must be an object")
        field_name = str(raw.get("field", "")).strip()
        if not field_name:
            raise ValueError("Each `ewsb.vevs` entry must define `field`")
        symbol = str(raw.get("symbol", raw.get("vev", ""))).strip()
        if not symbol:
            raise ValueError(f"EWSB VEV entry for {field_name!r} must define `symbol`")
        if field_name in seen_fields:
            raise ValueError(f"Duplicate EWSB VEV entry for field {field_name!r}")
        seen_fields.add(field_name)
        vevs.append({"field": field_name, "symbol": symbol})

    if not vevs:
        return None
    return {"vevs": tuple(vevs)}


def _section(title: str, body: str) -> str:
    return rf"""
\section{{{title}}}
{body}
"""


def _subsection(title: str, body: str) -> str:
    return rf"""
\subsection{{{title}}}
{body}
"""

def _dmath(body: str) -> str:
    return render_display_equations(body, separator=" " + "\\\\" + "\n")



def _join_terms(terms: list[str] | tuple[str, ...]) -> str:
    return " \\\\\n".join(term for term in terms if term and term != "0")

def _join_terms_grouped(terms: list[str] | tuple[str, ...], *, per_line: int) -> str:
    filtered = [term for term in terms if term and term != "0"]
    if per_line <= 0:
        raise ValueError("per_line must be positive")
    rows = [
        " + ".join(filtered[index : index + per_line])
        for index in range(0, len(filtered), per_line)
    ]
    return _join_terms(rows)


def _regroup_joined_terms(body: str, *, per_line: int) -> str:
    separator = " " + (chr(92) * 2) + chr(10)
    terms = [term.strip() for term in body.split(separator) if term.strip() and term.strip() != "0"]
    return _join_terms_grouped(terms, per_line=per_line)


def _join_categories(lagrangian, categories: list[str]) -> str:
    terms: list[str] = []
    for category in categories:
        terms.extend(operator.latex() for operator in lagrangian.by_category(category))
    return _join_terms(terms)


def _compact_terms(lagrangian, category: str) -> str:
    return _join_terms([operator.latex() for operator in lagrangian.by_category(category)])


def _expanded_terms(lagrangian, categories: list[str]) -> str:
    terms: list[str] = []
    category_set = set(categories)
    for operator in lagrangian.operators:
        if operator.category not in category_set:
            continue
        try:
            terms.append(operator.expanded_latex())
        except Exception as exc:
            terms.append(rf"\text{{Expansion unavailable for }} {operator.latex()} \quad \text{{({exc})}}")
    return " \\\\\n".join(terms)


def _compact_gauge_kinetic(lagrangian) -> str:
    return _join_terms_grouped(tuple(_kinetic_term_latex(term) for term in lagrangian.kinetic_terms), per_line=3)


def _expanded_gauge_kinetic(model) -> str:
    parts = [
        _join_terms_grouped(tuple(_kinetic_term_latex(term) for term in model.gauge_kinetic_terms), per_line=3),
        _regroup_joined_terms(model.gauge_interactions_latex(), per_line=3),
        _regroup_joined_terms(model.scalar_seagulls_latex(), per_line=3),
        _regroup_joined_terms(gauge_self_interactions_latex(), per_line=3),
    ]
    return _join_terms(parts)


def _kinetic_term_latex(term: str) -> str:
    table = {
        "-1/4 B_mn B^mn": r"-\frac{1}{4} B_{\mu\nu} B^{\mu\nu}",
        "-1/4 W_mn^a W^{a,mn}": r"-\frac{1}{4} W^a_{\mu\nu} W^{a,\mu\nu}",
        "-1/4 G_mn^A G^{A,mn}": r"-\frac{1}{4} G^A_{\mu\nu} G^{A,\mu\nu}",
    }
    if term in table:
        return table[term]
    if term.startswith("D[") and term.endswith("]† D[" + term[2:].split("]")[0] + "]"):
        field = term[2:].split("]")[0]
        field_latex = field_latex_name(field)
        return rf"(D_\mu {field_latex})^\dagger D^\mu {field_latex}"
    return rf"\mathrm{{{term.replace('_', r'\_')}}}"


def _su3_latex(su3: object) -> str:
    return su3.latex() if hasattr(su3, "latex") else str(su3)


def _field_line(
    *,
    name: str,
    kind: str,
    su3: object,
    su2: object,
    hypercharge: object,
    real: bool = False,
    colorize: bool = True,
) -> str:
    kind_text = "real scalar" if kind == "scalar" and real else kind.replace("_", " ")
    base_name = field_latex_name(name)
    if colorize:
        color = "blue" if kind == "scalar" else "red"
        display_name = rf"\textcolor{{{color}}}{{{base_name}}}"
    else:
        display_name = base_name
    return f"{display_name} & = & \\text{{{kind_text}}} & \\quad ({_su3_latex(su3)},{su2},{hypercharge})"


def _field_label(spec: FieldSpec) -> str:
    return _field_line(
        name=spec.name,
        kind=spec.kind,
        su3=spec.su3,
        su2=spec.su2,
        hypercharge=spec.hypercharge,
        real=spec.real,
    )


def _sm_field_label(field: Field) -> str:
    kind = "scalar" if field.kind == FieldKind.SCALAR else "weyl fermion"
    return _field_line(
        name=field.name,
        kind=kind,
        su3=field.su3,
        su2=field.su2,
        hypercharge=field.hypercharge,
        real=field.real,
        colorize=False,
    )


def _field_content_display(lines: tuple[str, ...]) -> str:
    return rf"""
\[
\begin{{aligned}}
{" \\\\" "\n".join(lines)}
\end{{aligned}}
\]
"""


def _anomaly_table(summary: str) -> str:
    text = summary.strip()
    if not text:
        return "\textbf{Anomalies:} unavailable"

    if ":" in text:
        status, details = text.split(":", 1)
        details = details.strip()
    else:
        status, details = text, ""

    is_free = status.strip() == "anomaly-free"
    status_word = "free" if is_free else "anomalous"
    status_color = "green!50!black" if is_free else "red"
    status_text = f"\\textcolor{{{status_color}}}{{\\textbf{{{status_word}}}}}"
    label_map = {
        "su3_cubic": "SU(3)c-SU(3)c-SU(3)c",
        "su2_su2_u1": "SU(2)L-SU(2)L-U(1)Y",
        "su3_su3_u1": "SU(3)c-SU(3)c-U(1)Y",
        "u1_gravity": "U(1)Y-gravity",
        "u1_cubic": "U(1)Y-U(1)Y-U(1)Y",
    }
    rows: list[tuple[str, str]] = []
    if details:
        for piece in details.split(","):
            item = piece.strip()
            if not item or "=" not in item:
                continue
            key, value = item.split("=", 1)
            label = label_map.get(key.strip(), _latex_escape_text(key.strip()))
            rows.append((label, _latex_escape_text(value.strip())))

    if not rows:
        return f"\\textbf{{Anomalies:}} {status_text}."

    table_rows = "\n".join(f"{label} & {value} \\\\" for label, value in rows)
    return (
        f"\\textbf{{Anomalies:}} {status_text}.\n"
        "\\[\n"
        "\\begin{array}{@{} l c @{}}\n"
        "\\text{Channel} & \\text{Value} \\\\" "\n"
        f"{table_rows}\n"
        "\\end{array}\n"
        "\\]\n"
    )


def _mixed_bsm_operator_texts(lagrangian, specs: tuple[FieldSpec, ...]) -> list[str]:
    names = {spec.name for spec in specs}
    lines: list[str] = []
    for operator in lagrangian.operators:
        present = {factor.field.name for factor in operator.factors}
        if len(present.intersection(names)) >= 2:
            lines.append(operator.text())
    return lines

def _mixed_bsm_operator_latex(lagrangian, specs: tuple[FieldSpec, ...]) -> str:
    names = {spec.name for spec in specs}
    terms: list[str] = []
    for operator in lagrangian.operators:
        present = {factor.field.name for factor in operator.factors}
        if len(present.intersection(names)) >= 2:
            terms.append(indexed_operator_latex(operator))
    return _join_terms(terms[:80])


def _latex_escape_text(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def _model_name_latex(model_name: str, specs: tuple[FieldSpec, ...]) -> str:
    aliases = _latex_alias_map(specs)
    parts: list[str] = []
    for chunk in model_name.split("+"):
        token = chunk.strip()
        if not token:
            continue
        if token == "SM":
            parts.append("SM")
            continue
        if any(spec.name == token for spec in specs):
            parts.append(rf"${aliases.get(token, field_latex_name(token))}$")
            continue
        parts.append(_latex_escape_text(token))
    return "+".join(parts) if parts else _latex_escape_text(model_name)


def _report_title_block(model_name: str, specs: tuple[FieldSpec, ...]) -> str:
    title = _model_name_latex(model_name, specs)
    return rf"""
\begin{{center}}
{{\LARGE \bfseries {title} \par}}
\vspace{{0.35em}}
{{\large Model Report \par}}
\vspace{{0.7em}}
\fcolorbox{{black}}{{gray!10}}{{\textsf{{\small Generated by bsm\_agent || Shaikh Saad}}}}
\end{{center}}
\vspace{{1em}}
"""


def _render_report(
    model_name: str,
    specs: tuple[FieldSpec, ...],
    model,
    lagrangian,
    *,
    expand_sectors: tuple[str, ...] = REPORT_EXPANSION_SECTORS,
    ewsb_config: dict | None = None,
) -> str:
    sm_fields = StandardModel().fields
    sm_field_block = _field_content_display(tuple(_sm_field_label(field) for field in sm_fields))
    field_alias_map = _latex_alias_map(specs)
    report_alias_map = _report_field_latex_alias_map(specs)
    with field_latex_aliases(field_alias_map):
        field_block = _field_content_display(tuple(_field_label(spec) for spec in specs)) if specs else ""
    with field_latex_aliases(report_alias_map):
        anomaly_block = _anomaly_table(model.anomaly_report().summary())
        compact = "\n".join(
            [
                _subsection("Yukawa and Fermion-Mass Terms", _dmath(_join_categories(lagrangian, ["yukawa", "fermion_mass"]))),
                _subsection("Scalar Potential", _dmath(_compact_terms(lagrangian, "scalar_potential"))),
                _subsection("Gauge and Kinetic Terms", _dmath(_compact_gauge_kinetic(lagrangian))),
            ]
        )
        expand_set = set(expand_sectors)
        detailed_body = ""
        if expand_set:
            detailed = "\n".join(
                [
                    _subsection(
                        "Yukawa and Fermion-Mass Terms",
                        _dmath(
                            _expanded_terms(lagrangian, ["yukawa", "fermion_mass"])
                            if "yukawa" in expand_set
                            else _join_categories(lagrangian, ["yukawa", "fermion_mass"])
                        ),
                    ),
                    _subsection(
                        "Scalar Potential",
                        _dmath(
                            _expanded_terms(lagrangian, ["scalar_potential"])
                            if "scalar_potential" in expand_set
                            else _compact_terms(lagrangian, "scalar_potential")
                        ),
                    ),
                    _subsection(
                        "Gauge and Kinetic Terms",
                        _dmath(
                            _expanded_gauge_kinetic(model)
                            if "gauge_kinetic" in expand_set
                            else _compact_gauge_kinetic(lagrangian)
                        ),
                    ),
                ]
            )
            detailed_title = (
                "Fully Expanded Form of the Lagrangian"
                if expand_set == set(REPORT_EXPANSION_SECTORS)
                else "Detailed Form of the Lagrangian"
            )
            detailed_body = _section(detailed_title, detailed)
        ewsb_section = _ewsb_report_section(model, lagrangian, ewsb_config)
        ewsb_body = "" if ewsb_section is None else _section("EWSB Mass Matrices", ewsb_section)

    overview_pieces = [
        _subsection("Standard Model Field Content", sm_field_block),
    ]
    if specs:
        overview_pieces.append(_subsection("BSM Field Content", field_block))
    overview_pieces.append(_subsection("Anomaly Summary", anomaly_block))
    overview = "\n".join(overview_pieces)

    return rf"""\documentclass[11pt]{{article}}
\usepackage[a4paper,margin=0.7in]{{geometry}}
\usepackage{{amsmath,amssymb,breqn}}
\usepackage{{xcolor}}
\usepackage[colorlinks=true,linkcolor=blue,urlcolor=blue]{{hyperref}}
\begin{{document}}
{_report_title_block(model_name, specs)}
\tableofcontents
\bigskip

{_section("Field Content and Anomalies", overview)}

{_section("Compact Form of the Full Lagrangian", compact)}

{detailed_body}

{ewsb_body}

\end{{document}}
"""


def _ewsb_report_section(model, lagrangian, ewsb_config: dict | None) -> str | None:
    if ewsb_config is None:
        return None

    try:
        vev_fields = _resolve_ewsb_vev_fields(model, ewsb_config)
    except ValueError as exc:
        detail = str(exc).replace("_", r"\_")
        return rf"""\[
\text{{{detail}}}
\]"""

    vacuum_substitutions = _combine_vev_substitutions(vev_fields, shifts=False)
    vev_shifts = _combine_vev_substitutions(vev_fields, shifts=True)
    masses = compute_mass_matrices(lagrangian, vev_shifts)
    stationary_condition = _stationary_conditions_latex(lagrangian, vev_fields, vacuum_substitutions)

    neutral_labels = _neutral_scalar_component_labels(model)
    remaining_hermitian = _exclude_scalar_labels(masses.scalar_hermitian_blocks, neutral_labels)
    remaining_holomorphic = _exclude_scalar_labels(masses.scalar_holomorphic_blocks, neutral_labels)

    pieces = [
        _subsection("VEV Substitutions", _dmath(_vev_substitutions_latex(masses.vev_substitutions))),
        _subsection("Stationary Condition", _dmath(stationary_condition)),
        _subsection("Neutral Scalar Mass Matrix", _neutral_scalar_mass_matrix_latex(model, lagrangian, vev_shifts)),
        _subsection("Scalar Hermitian Blocks", _matrix_blocks_latex(remaining_hermitian, power=True)),
        _subsection("Scalar Holomorphic Blocks", _matrix_blocks_latex(remaining_holomorphic, power=False)),
        _subsection("Fermion Blocks", _matrix_blocks_latex(masses.fermion_blocks, power=False)),
    ]
    return "\n".join(pieces)


def _resolve_ewsb_vev_fields(model, ewsb_config: dict) -> tuple[tuple[Field, sp.Symbol], ...]:
    vev_entries = ewsb_config.get("vevs")
    if not vev_entries:
        if "sm_higgs_vev" in ewsb_config:
            vev_entries = ({"field": "H", "symbol": ewsb_config["sm_higgs_vev"]},)
        else:
            vev_entries = ()

    resolved: list[tuple[Field, sp.Symbol]] = []
    for entry in vev_entries:
        field_name = entry["field"]
        field = next((candidate for candidate in model.fields if candidate.name == field_name), None)
        if field is None:
            raise ValueError(f"EWSB configuration requested, but the model has no field {field_name}.")
        if field.kind != FieldKind.SCALAR:
            raise ValueError(f"EWSB VEV field {field_name} is not a scalar.")
        if field.su3.dimension != 1:
            raise ValueError(f"EWSB VEV field {field_name} is not color-singlet.")
        try:
            component_label(field, 0)
        except ValueError as exc:
            raise ValueError(f"EWSB VEV field {field_name} has no neutral component.") from exc
        resolved.append((field, sp.Symbol(entry["symbol"], real=True)))
    return tuple(resolved)


def _combine_vev_substitutions(
    vev_fields: tuple[tuple[Field, sp.Symbol], ...],
    *,
    shifts: bool,
) -> dict[str, sp.Expr | tuple[sp.Expr, bool]]:
    substitutions: dict[str, sp.Expr | tuple[sp.Expr, bool]] = {}
    for field, symbol in vev_fields:
        current = neutral_scalar_vev_shifts(field, symbol) if shifts else neutral_scalar_vev_substitutions(field, symbol)
        overlap = set(substitutions).intersection(current)
        if overlap:
            names = ", ".join(sorted(overlap))
            raise ValueError(f"Duplicate EWSB VEV substitution for: {names}")
        substitutions.update(current)
    return substitutions


def _stationary_conditions_latex(
    lagrangian,
    vev_fields: tuple[tuple[Field, sp.Symbol], ...],
    vev_substitutions: dict[str, sp.Expr],
) -> str:
    vacuum_potential = sp.Integer(0)
    for operator in lagrangian.by_category("scalar_potential"):
        coefficient = sp.Symbol(operator.coefficient)
        for term in expand_operator(operator):
            expr = sp.simplify(coefficient * term.coefficient)
            for field in term.fields:
                value = vev_substitutions.get(field)
                if value is None:
                    expr = sp.Integer(0)
                    break
                expr = sp.simplify(expr * value)
            vacuum_potential += expr
            if operator.add_hc:
                vacuum_potential += sp.conjugate(expr)

    conditions = [
        rf"\frac{{\partial V}}{{\partial {latex_identifier(symbol.name)}}} = {_expr_latex(sp.simplify(sp.diff(sp.expand(vacuum_potential), symbol)))} = 0"
        for _, symbol in vev_fields
    ]
    return (" " + chr(92) + chr(92) + "\n").join(conditions)


def _vev_substitutions_latex(substitutions: dict[str, sp.Expr | tuple[sp.Expr, bool]]) -> str:
    if not substitutions:
        return r"\text{No VEV substitutions provided.}"

    pieces: list[str] = []
    for lhs, rhs in substitutions.items():
        if isinstance(rhs, tuple):
            shift, keep_field = rhs
            target = rf"{lhs} + {sp.latex(shift)}" if keep_field else sp.latex(shift)
        else:
            target = sp.latex(rhs)
        pieces.append(rf"{lhs} &\to {target}")
    separator = " " + chr(92) + chr(92) + "\n"
    return separator.join(pieces)


def _neutral_scalar_component_labels(model) -> tuple[str, ...]:
    labels: list[str] = []
    for field in model.fields:
        if field.kind != FieldKind.SCALAR or field.su3.dimension != 1:
            continue
        try:
            labels.append(component_label(field, 0))
        except ValueError:
            continue
    return tuple(labels)



def _exclude_scalar_labels(blocks, excluded: tuple[str, ...]):
    excluded_set = set(excluded)
    filtered = []
    for block in blocks:
        row_indices = [index for index, label in enumerate(block.row_fields) if label not in excluded_set]
        col_indices = [index for index, label in enumerate(block.column_fields) if label not in excluded_set]
        if not row_indices or not col_indices:
            continue
        matrix = sp.Matrix([
            [sp.simplify(block.matrix[row, col]) for col in col_indices]
            for row in row_indices
        ])
        if all(entry == 0 for entry in matrix):
            continue
        filtered.append(
            type(block)(
                kind=block.kind,
                row_fields=tuple(block.row_fields[index] for index in row_indices),
                column_fields=tuple(block.column_fields[index] for index in col_indices),
                matrix=matrix,
            )
        )
    return tuple(filtered)



def _neutral_scalar_mass_matrix_latex(model, lagrangian, vev_shifts: dict[str, sp.Expr | tuple[sp.Expr, bool]]) -> str:
    complex_labels: list[str] = []
    real_labels: list[str] = []
    toggle: dict[str, str] = {}
    tracked_symbols: dict[str, sp.Symbol] = {}
    substitutions: dict[sp.Symbol, sp.Expr] = {}
    even_symbols: list[sp.Symbol] = []
    even_labels: list[str] = []
    odd_symbols: list[sp.Symbol] = []
    odd_labels: list[str] = []

    for field in model.fields:
        if field.kind != FieldKind.SCALAR or field.su3.dimension != 1:
            continue
        try:
            neutral = component_label(field, 0)
        except ValueError:
            continue

        if field.real:
            symbol = sp.Symbol(f"neutral_real_{len(real_labels)}", real=True)
            real_labels.append(neutral)
            tracked_symbols[neutral] = symbol
            toggle[neutral] = neutral
            even_symbols.append(symbol)
            even_labels.append(neutral)
            continue

        dagger = component_label(field, 0, conjugate=True)
        re_symbol = sp.Symbol(f"neutral_re_{len(complex_labels)}", real=True)
        im_symbol = sp.Symbol(f"neutral_im_{len(complex_labels)}", real=True)
        complex_symbol = sp.Symbol(f"neutral_complex_{len(complex_labels)}")
        dagger_symbol = sp.Symbol(f"neutral_complex_{len(complex_labels)}_dagger")
        complex_labels.append(neutral)
        tracked_symbols[neutral] = complex_symbol
        tracked_symbols[dagger] = dagger_symbol
        toggle[neutral] = dagger
        toggle[dagger] = neutral
        substitutions[complex_symbol] = (re_symbol + sp.I * im_symbol) / sp.sqrt(2)
        substitutions[dagger_symbol] = (re_symbol - sp.I * im_symbol) / sp.sqrt(2)
        even_symbols.append(re_symbol)
        even_labels.append(rf"\operatorname{{Re}}({neutral})")
        odd_symbols.append(im_symbol)
        odd_labels.append(rf"\operatorname{{Im}}({neutral})")

    basis_symbols = tuple(even_symbols + odd_symbols)
    basis_labels = tuple(even_labels + odd_labels)
    if not basis_symbols:
        return r"""\[
\text{No neutral scalar fields.}
\]"""

    shifted = {str(label): _report_substitution_value(value) for label, value in vev_shifts.items()}
    quadratic = sp.Integer(0)
    for operator in lagrangian.by_category("scalar_potential"):
        coefficient = sp.Symbol(operator.coefficient)
        for term in expand_operator(operator):
            contributions = [(sp.simplify(coefficient * term.coefficient), [])]
            for field in term.fields:
                next_contributions: list[tuple[sp.Expr, list[str]]] = []
                for expr, remaining in contributions:
                    for factor, survivor in _report_substitution_contributions(field, shifted):
                        shifted_expr = sp.simplify(expr * factor)
                        if shifted_expr == 0:
                            continue
                        next_remaining = list(remaining)
                        if survivor is not None:
                            next_remaining.append(survivor)
                        next_contributions.append((shifted_expr, next_remaining))
                contributions = next_contributions

            for expr, remaining in contributions:
                if len(remaining) != 2 or expr == 0 or not all(label in tracked_symbols for label in remaining):
                    continue
                quadratic += expr * tracked_symbols[remaining[0]] * tracked_symbols[remaining[1]]
                if operator.add_hc:
                    hc_fields = tuple(toggle[label] for label in reversed(remaining))
                    quadratic += sp.conjugate(expr) * tracked_symbols[hc_fields[0]] * tracked_symbols[hc_fields[1]]

    quadratic = sp.expand(quadratic.subs(substitutions))
    matrix = sp.Matrix([
        [sp.simplify(sp.diff(quadratic, left, right)) for right in basis_symbols]
        for left in basis_symbols
    ])

    basis_lines = [
        rf"b_{{{index}}} = {label}"
        for index, label in enumerate(basis_labels, start=1)
    ]
    entry_lines = [
        rf"M^2_{{\text{{neutral}},{row_index}{col_index}}} = {_expr_latex(matrix[row_index - 1, col_index - 1])}"
        for row_index in range(1, matrix.rows + 1)
        for col_index in range(1, matrix.cols + 1)
    ]
    separator = " " + chr(92) + chr(92) + "\n"
    return _dmath(separator.join([
        r"\text{Neutral basis}",
        *basis_lines,
        r"\text{Neutral scalar mass-matrix entries}",
        *entry_lines,
    ]))


def _report_substitution_value(value: sp.Expr | int | tuple[sp.Expr | int, bool]) -> sp.Expr | tuple[sp.Expr, bool]:
    if isinstance(value, tuple):
        shift, keep_field = value
        return (sp.sympify(shift), bool(keep_field))
    return sp.sympify(value)



def _report_substitution_contributions(
    field: str,
    substitutions: dict[str, sp.Expr | tuple[sp.Expr, bool]],
):
    value = substitutions.get(field)
    if value is None:
        return ((sp.Integer(1), field),)
    if isinstance(value, tuple):
        shift, keep_field = value
        if keep_field:
            return ((sp.Integer(1), field), (shift, None))
        return ((shift, None),)
    return ((value, None),)

def _conjugate_basis_label(label: str) -> str:
    return rf"\left({label}\right)^\dagger"


def _expr_latex(expr: sp.Expr) -> str:
    latex = sp.latex(expr)
    for symbol in sorted(expr.free_symbols, key=lambda item: len(item.name), reverse=True):
        latex = latex.replace(sp.latex(symbol), latex_identifier(symbol.name))
    return latex


def _matrix_blocks_latex(blocks, *, power: bool) -> str:
    if not blocks:
        return r"""\[
\text{No blocks generated.}
\]"""

    lines: list[str] = []
    for index, block in enumerate(blocks, start=1):
        display_rows = block.row_fields
        if block.kind == "scalar_hermitian":
            display_rows = tuple(_conjugate_basis_label(label) for label in block.row_fields)
        rows = ", ".join(display_rows)
        cols = ", ".join(block.column_fields)
        symbol = "M^2" if power else "M"
        label = rf"{symbol}_{{{index}}}"
        matrix_latex = _expr_latex(block.matrix)
        lines.append(
            rf"""
\[
\begin{{array}}{{l}}
{label}[{rows};{cols}] = \\
{matrix_latex}
\end{{array}}
\]
"""
        )
    return "\n".join(lines)


def _write_pdf(tex_path: Path) -> Path | None:
    if shutil.which("pdflatex") is None:
        return None
    result = None
    for _ in range(2):
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", tex_path.name],
            cwd=tex_path.parent,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    pdf_path = tex_path.with_suffix(".pdf")
    if result is not None and result.returncode != 0 and not pdf_path.exists():
        raise RuntimeError(f"pdflatex failed; see {tex_path.with_suffix('.log')}")
    return pdf_path
