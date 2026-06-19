"""Lagrangian operators and formatting helpers."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from typing import Iterable

from .fields import FieldFactor, FieldKind


def latex_identifier(name: str) -> str:
    parts = name.split("_")
    greek = {
        "alpha": r"\alpha",
        "beta": r"\beta",
        "gamma": r"\gamma",
        "delta": r"\delta",
        "epsilon": r"\epsilon",
        "lambda": r"\lambda",
        "mu": r"\mu",
        "nu": r"\nu",
        "rho": r"\rho",
        "sigma": r"\sigma",
        "theta": r"\theta",
        "xi": r"\xi",
        "zeta": r"\zeta",
    }
    special = {
        "m2": r"m^{2}",
        "xi3": r"\xi^{3}",
    }
    head = special.get(parts[0], greek.get(parts[0], parts[0]))
    if len(parts) == 1:
        return head
    return rf"{head}_{{{','.join(parts[1:])}}}"


@dataclass(frozen=True)
class Operator:
    factors: tuple[FieldFactor, ...]
    category: str
    coefficient: str
    add_hc: bool = False
    contraction_index: int = 1
    contraction_count: int = 1
    phase_target_fields: tuple[str, ...] = ()
    dual_basis_slots: tuple[int, ...] = ()

    @property
    def dimension(self) -> Fraction:
        return sum((factor.mass_dimension for factor in self.factors), Fraction(0))

    def signature(self) -> tuple:
        return tuple((f.field.name, f.conjugate) for f in self.factors)

    def field_counts(self) -> Counter[str]:
        return Counter(f.name for f in self.factors)

    def text(self) -> str:
        body = " ".join(f.name for f in self.factors)
        contraction = f" [contract {self.contraction_index}/{self.contraction_count}]" if self.contraction_count > 1 else ""
        suffix = " + h.c." if self.add_hc else ""
        return f"{self.coefficient} {body}{contraction}{suffix}"

    def latex(self) -> str:
        body = r" ".join(f.latex() for f in self.factors)
        suffix = r" + \mathrm{h.c.}" if self.add_hc else ""
        return rf"{latex_identifier(self.coefficient)} \, {body}{suffix}"

    def expanded_latex(self) -> str:
        return _expanded_operator_latex_cached(self)


@dataclass(frozen=True)
class Lagrangian:
    kinetic_terms: tuple[str, ...]
    operators: tuple[Operator, ...]

    def by_category(self, category: str) -> list[Operator]:
        return [op for op in self.operators if op.category == category]

    def summary(self) -> str:
        counts = Counter(op.category for op in self.operators)
        lines = ["Lagrangian summary:"]
        lines.append(f"  kinetic terms: {len(self.kinetic_terms)}")
        for category in sorted(counts):
            lines.append(f"  {category}: {counts[category]}")
        return "\n".join(lines)

    def text(self) -> str:
        parts = [*self.kinetic_terms, *(op.text() for op in self.operators)]
        return "\n".join(parts)

    def latex(self) -> str:
        terms = [r"\mathcal{L}_{\mathrm{gauge+kinetic}}"]
        terms.extend(op.latex() for op in self.operators)
        return " \\\\\n".join(terms)

    def expanded_latex(self, *, categories: Iterable[str] | None = None) -> str:
        selected = self.operators
        if categories is not None:
            category_set = set(categories)
            selected = tuple(op for op in selected if op.category in category_set)
        terms: list[str] = []
        for operator in selected:
            try:
                terms.append(operator.expanded_latex())
            except NotImplementedError as exc:
                terms.append(rf"\text{{Expansion unavailable for }} {operator.latex()} \quad \text{{({exc})}}")
        return " \\\\\n".join(terms)


def has_fermions(factors: Iterable[FieldFactor], count: int) -> bool:
    return sum(f.kind == FieldKind.WEYL_FERMION for f in factors) == count


def has_scalars(factors: Iterable[FieldFactor], count: int) -> bool:
    return sum(f.kind == FieldKind.SCALAR for f in factors) == count


def _expanded_operator_latex_cached(operator: Operator) -> str:
    from .expansion import expanded_operator_latex

    return _expanded_operator_latex_cached_impl(operator)


@lru_cache(maxsize=None)
def _expanded_operator_latex_cached_impl(operator: Operator) -> str:
    from .expansion import expanded_operator_latex

    return expanded_operator_latex(operator)
