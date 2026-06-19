"""Mass-matrix extraction from the symbolic operator basis.

This module expands operators into gauge-eigenstate components, substitutes
declared VEVs, and collects the quadratic bilinears that define tree-level mass
matrices.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
import re

import sympy as sp

from .expansion import _component_latex, _weights, expand_operator
from .fields import Field, FieldKind
from .operators import Lagrangian


@dataclass(frozen=True)
class MassMatrixBlock:
    kind: str
    row_fields: tuple[str, ...]
    column_fields: tuple[str, ...]
    matrix: sp.Matrix


@dataclass(frozen=True)
class MassMatrixResult:
    vev_substitutions: dict[str, sp.Expr]
    scalar_hermitian_entries: dict[tuple[str, str], sp.Expr]
    scalar_holomorphic_entries: dict[tuple[str, str], sp.Expr]
    fermion_entries: dict[tuple[str, str], sp.Expr]
    scalar_hermitian_blocks: tuple[MassMatrixBlock, ...]
    scalar_holomorphic_blocks: tuple[MassMatrixBlock, ...]
    fermion_blocks: tuple[MassMatrixBlock, ...]

    def scalar_hermitian_mass(self, left: str, right: str) -> sp.Expr:
        return self.scalar_hermitian_entries.get((_normalize_component_label(left), _normalize_component_label(right)), sp.Integer(0))

    def scalar_holomorphic_mass(self, left: str, right: str) -> sp.Expr:
        return self.scalar_holomorphic_entries.get((_normalize_component_label(left), _normalize_component_label(right)), sp.Integer(0))

    def fermion_mass(self, left: str, right: str) -> sp.Expr:
        return self.fermion_entries.get((_normalize_component_label(left), _normalize_component_label(right)), sp.Integer(0))


def component_label(
    field: Field,
    electric_charge: Fraction | int | float | str,
    *,
    conjugate: bool = False,
    color: int | tuple[int, int] | None = None,
) -> str:
    """Return the displayed component label used by the expansion backend."""

    if field.kind == FieldKind.GAUGE_BOSON:
        raise ValueError("Gauge-boson component labels are not supported here")
    charge = Fraction(electric_charge)
    factor = field.factor(conjugate)
    for two_m in _weights(field.su2):
        display_two_m = -two_m if conjugate else two_m
        if Fraction(display_two_m, 2) + field.hypercharge == charge:
            return _component_latex(factor, two_m, color)
    raise ValueError(f"Field {field.name} has no component with electric charge {charge}")


def neutral_scalar_vev_substitutions(field: Field, vev: sp.Expr | int) -> dict[str, sp.Expr]:
    """Return the vacuum-value map for the neutral scalar components of a field."""

    if field.kind != FieldKind.SCALAR:
        raise ValueError(f"{field.name} is not a scalar field")
    if field.su3.dimension != 1:
        raise ValueError(f"{field.name} is not color-singlet and cannot receive the SM-like VEV here")

    neutral = component_label(field, 0)
    vev_expr = sp.sympify(vev)
    if field.real:
        return {neutral: vev_expr}

    normalized = sp.simplify(vev_expr / sp.sqrt(2))
    return {
        neutral: normalized,
        component_label(field, 0, conjugate=True): normalized,
    }


def neutral_scalar_vev_shifts(field: Field, vev: sp.Expr | int) -> dict[str, tuple[sp.Expr, bool]]:
    """Return the standard neutral-scalar field shifts used for EWSB mass matrices."""

    shifts = neutral_scalar_vev_substitutions(field, vev)
    return {label: (value, True) for label, value in shifts.items()}


def compute_mass_matrices(
    lagrangian: Lagrangian,
    vev_substitutions: Mapping[str, sp.Expr | int | tuple[sp.Expr | int, bool]],
) -> MassMatrixResult:
    r"""Extract scalar and fermion mass matrices after substituting VEVs.

    The returned scalar matrices are split into hermitian bilinears
    ``phi_i^\dagger phi_j`` and holomorphic bilinears ``phi_i phi_j``.  This
    keeps the result explicit in the component basis already used elsewhere in
    the package.
    """

    vevs = {str(label): _normalize_substitution_value(value) for label, value in vev_substitutions.items()}
    scalar_hermitian: defaultdict[tuple[str, str], sp.Expr] = defaultdict(lambda: sp.Integer(0))
    scalar_holomorphic: defaultdict[tuple[str, str], sp.Expr] = defaultdict(lambda: sp.Integer(0))
    fermion_entries: defaultdict[tuple[str, str], sp.Expr] = defaultdict(lambda: sp.Integer(0))

    for operator in lagrangian.operators:
        if operator.category not in {"scalar_potential", "fermion_mass", "yukawa"}:
            continue
        coefficient = sp.Symbol(operator.coefficient)
        for term in expand_operator(operator):
            contributions = [(sp.simplify(coefficient * term.coefficient), [])]
            for field in term.fields:
                next_contributions: list[tuple[sp.Expr, list[str]]] = []
                for expr, remaining in contributions:
                    for factor, survivor in _substitution_contributions(field, vevs):
                        shifted = sp.simplify(expr * factor)
                        if shifted == 0:
                            continue
                        next_remaining = list(remaining)
                        if survivor is not None:
                            next_remaining.append(survivor)
                        next_contributions.append((shifted, next_remaining))
                contributions = next_contributions

            for expr, remaining in contributions:
                if len(remaining) != 2 or expr == 0:
                    continue

                if operator.category == "scalar_potential":
                    _accumulate_scalar_entry(
                        scalar_hermitian,
                        scalar_holomorphic,
                        remaining[0],
                        remaining[1],
                        expr,
                        include_hc=operator.add_hc,
                    )
                    continue

                left = _normalize_component_label(remaining[0])
                right = _normalize_component_label(remaining[1])
                fermion_entries[(left, right)] += expr

    scalar_hermitian_entries = _freeze_entries(scalar_hermitian)
    scalar_holomorphic_entries = _freeze_entries(scalar_holomorphic)
    fermion_entries_frozen = _freeze_entries(fermion_entries)

    return MassMatrixResult(
        vev_substitutions=dict(vevs),
        scalar_hermitian_entries=scalar_hermitian_entries,
        scalar_holomorphic_entries=scalar_holomorphic_entries,
        fermion_entries=fermion_entries_frozen,
        scalar_hermitian_blocks=_blocks_from_entries("scalar_hermitian", scalar_hermitian_entries),
        scalar_holomorphic_blocks=_blocks_from_entries("scalar_holomorphic", scalar_holomorphic_entries),
        fermion_blocks=_blocks_from_entries("fermion", fermion_entries_frozen),
    )


def _normalize_substitution_value(value: sp.Expr | int | tuple[sp.Expr | int, bool]) -> sp.Expr | tuple[sp.Expr, bool]:
    if isinstance(value, tuple):
        shift, keep_field = value
        return (sp.sympify(shift), bool(keep_field))
    return sp.sympify(value)


def _substitution_contributions(
    field: str,
    substitutions: Mapping[str, sp.Expr | tuple[sp.Expr, bool]],
) -> Sequence[tuple[sp.Expr, str | None]]:
    value = substitutions.get(field)
    if value is None:
        return ((sp.Integer(1), field),)
    if isinstance(value, tuple):
        shift, keep_field = value
        if keep_field:
            return ((sp.Integer(1), field), (shift, None))
        return ((shift, None),)
    return ((value, None),)


def _freeze_entries(entries: Mapping[tuple[str, str], sp.Expr]) -> dict[tuple[str, str], sp.Expr]:
    return {
        key: sp.simplify(value)
        for key, value in sorted(entries.items())
        if value != 0
    }


def _accumulate_scalar_entry(
    hermitian: defaultdict[tuple[str, str], sp.Expr],
    holomorphic: defaultdict[tuple[str, str], sp.Expr],
    left: str,
    right: str,
    expr: sp.Expr,
    *,
    include_hc: bool,
) -> None:
    left_dagger = _is_daggered(left)
    right_dagger = _is_daggered(right)
    left_label = _normalize_component_label(left)
    right_label = _normalize_component_label(right)

    if left_dagger and not right_dagger:
        hermitian[(left_label, right_label)] += expr
        if include_hc:
            hermitian[(right_label, left_label)] += sp.conjugate(expr)
        return

    if not left_dagger and right_dagger:
        hermitian[(right_label, left_label)] += expr
        if include_hc:
            hermitian[(left_label, right_label)] += sp.conjugate(expr)
        return

    if not left_dagger and not right_dagger:
        holomorphic[(left_label, right_label)] += expr
        return

    if include_hc:
        holomorphic[(left_label, right_label)] += sp.conjugate(expr)


def _blocks_from_entries(kind: str, entries: Mapping[tuple[str, str], sp.Expr]) -> tuple[MassMatrixBlock, ...]:
    if not entries:
        return ()

    adjacency: dict[str, set[str]] = defaultdict(set)
    nodes: set[str] = set()
    for left, right in entries:
        adjacency[left].add(right)
        adjacency[right].add(left)
        nodes.add(left)
        nodes.add(right)

    blocks: list[MassMatrixBlock] = []
    seen: set[str] = set()
    for start in sorted(nodes):
        if start in seen:
            continue
        stack = [start]
        component: set[str] = set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            seen.add(node)
            stack.extend(sorted(adjacency[node] - component))

        rows = tuple(sorted(label for label in component if any((label, other) in entries for other in component)))
        cols = tuple(sorted(label for label in component if any((other, label) in entries for other in component)))
        matrix = sp.Matrix(
            [
                [sp.simplify(entries.get((row, col), sp.Integer(0))) for col in cols]
                for row in rows
            ]
        )
        blocks.append(MassMatrixBlock(kind=kind, row_fields=rows, column_fields=cols, matrix=matrix))
    return tuple(blocks)


def _is_daggered(label: str) -> bool:
    return label.endswith(r"^\dagger")


def _normalize_component_label(label: str) -> str:
    if _is_daggered(label):
        label = _canonicalize_daggered_component_label(label[: -len(r"^\dagger")])
    return _strip_outer_braces(label)


def _canonicalize_daggered_component_label(label: str) -> str:
    text = _strip_outer_braces(label)
    text = re.sub(
        r'\^\{([^{}]+),\\bar\{([^{}]+)\}\}',
        lambda match: f'^{{{match.group(1)}}}_{{{match.group(2)}}}',
        text,
    )
    text = re.sub(
        r'\^\{([^{}]+),\\overline\{([^{}]+)\}\}',
        lambda match: f'^{{{match.group(1)}}}_{{{match.group(2)}}}',
        text,
    )
    return text


def _strip_outer_braces(text: str) -> str:
    value = text
    while value.startswith("{") and value.endswith("}") and _balanced_outer_braces(value):
        value = value[1:-1]
    return value


def _balanced_outer_braces(text: str) -> bool:
    depth = 0
    for index, char in enumerate(text):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and index != len(text) - 1:
                return False
        if depth < 0:
            return False
    return depth == 0
