"""Gauge-interaction expansion from covariant derivatives."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import sympy as sp

from .expansion import (
    _cg,
    _color_suffix,
    _component_latex,
    _weights,
)
from .fields import Field, FieldKind
from .groups import SU2Rep, SU3Rep
from .su3_invariants import color_labels as _su3_color_labels
from .su3_invariants import generators as _general_su3_generators
from .su3_invariants import _su3_f_constants


def _prefactor_latex(coeff: sp.Expr, couplings: list[str]) -> list[str]:
    coeff = sp.simplify(coeff)
    coupling_block = r"\, ".join(couplings)
    if coeff == 1:
        return [coupling_block]
    if coeff == -1:
        return ["-", coupling_block]
    coeff_latex = sp.latex(coeff)
    if coeff.could_extract_minus_sign():
        return [rf"\left({coeff_latex}\right)", coupling_block]
    return [coeff_latex, coupling_block]


@dataclass(frozen=True)
class GaugeInteractionTerm:
    coefficient: sp.Expr
    coupling: str
    gauge_boson: str
    left_component: str
    right_component: str
    current: str

    def latex(self) -> str:
        pieces = _prefactor_latex(self.coefficient, [self.coupling])
        pieces.extend([self.gauge_boson, self.left_component, self.current, self.right_component])
        return r"\, ".join(pieces)


@dataclass(frozen=True)
class SeagullTerm:
    coefficient: sp.Expr
    coupling_left: str
    coupling_right: str
    gauge_left: str
    gauge_right: str
    left_component: str
    right_component: str

    def latex(self) -> str:
        pieces = _prefactor_latex(self.coefficient, [self.coupling_left, self.coupling_right])
        pieces.extend([self.gauge_left, self.gauge_right, self.left_component, self.right_component])
        return r"\, ".join(pieces)


@dataclass(frozen=True)
class GaugeSelfInteractionTerm:
    coefficient: sp.Expr
    latex_body: str

    def latex(self) -> str:
        coeff = sp.simplify(self.coefficient)
        if coeff == 1:
            return self.latex_body
        if coeff == -1:
            return "- " + self.latex_body
        return rf"{sp.latex(coeff)}\, {self.latex_body}"


def gauge_interactions_latex(field: Field) -> str:
    terms = gauge_interaction_terms(field)
    if not terms:
        return "0"
    return " \\\\\n".join(term.latex() for term in terms)


def scalar_seagull_latex(field: Field) -> str:
    terms = scalar_seagull_terms(field)
    if not terms:
        return "0"
    return " \\\\\n".join(term.latex() for term in terms)


def gauge_self_interactions_latex() -> str:
    return " \\\\\n".join(term.latex() for term in gauge_self_interaction_terms())


def gauge_interaction_terms(field: Field) -> list[GaugeInteractionTerm]:
    terms: list[GaugeInteractionTerm] = []
    terms.extend(_u1_terms(field))
    terms.extend(_su2_terms(field))
    terms.extend(_su3_terms(field))
    return terms


def scalar_seagull_terms(field: Field) -> list[SeagullTerm]:
    if field.kind != FieldKind.SCALAR:
        return []
    generator_sets = _gauge_generator_sets(field)
    terms: list[SeagullTerm] = []
    for left_set in generator_sets:
        for right_set in generator_sets:
            for left_label, left_matrix in left_set["generators"]:
                for right_label, right_matrix in right_set["generators"]:
                    product = left_matrix * right_matrix
                    terms.extend(_matrix_seagull_terms(field, left_set, right_set, left_label, right_label, product))
    return terms


def gauge_self_interaction_terms() -> list[GaugeSelfInteractionTerm]:
    terms: list[GaugeSelfInteractionTerm] = []
    terms.extend(_nonabelian_self_terms("g_2", "W", _su2_f_constants()))
    terms.extend(_nonabelian_self_terms("g_3", "G", _su3_f_constants()))
    return terms


def _current_symbol(field: Field) -> str:
    if field.kind == FieldKind.SCALAR:
        return r"i\overleftrightarrow{\partial^\mu}"
    if field.kind == FieldKind.WEYL_FERMION:
        return r"\bar\sigma^\mu"
    return r"J^\mu"


def _u1_terms(field: Field) -> list[GaugeInteractionTerm]:
    if field.hypercharge == 0:
        return []
    terms = []
    for weak in _weak_labels(field.su2):
        for color in _color_labels(field.su3):
            component = _component_latex(field.factor(False), weak, color)
            dagger = _component_latex(field.factor(True), weak, color, conjugate_dual_weight=False)
            terms.append(
                GaugeInteractionTerm(
                    coefficient=sp.Rational(field.hypercharge.numerator, field.hypercharge.denominator),
                    coupling="g_1",
                    gauge_boson=r"B_\mu",
                    left_component=dagger,
                    right_component=component,
                    current=_current_symbol(field),
                )
            )
    return terms


def _su2_terms(field: Field) -> list[GaugeInteractionTerm]:
    if field.su2 == SU2Rep(1):
        return []
    terms = []
    matrices = _su2_generators(field.su2)
    weights = _weak_labels(field.su2)
    for adjoint_label, matrix in enumerate(matrices, start=1):
        for row, left_weight in enumerate(weights):
            for col, right_weight in enumerate(weights):
                coeff = sp.simplify(matrix[row, col])
                if coeff == 0:
                    continue
                for color in _color_labels(field.su3):
                    terms.append(
                        GaugeInteractionTerm(
                            coefficient=coeff,
                            coupling="g_2",
                            gauge_boson=rf"W^{{{adjoint_label}}}_\mu",
                            left_component=_component_latex(field.factor(True), left_weight, color, conjugate_dual_weight=False),
                            right_component=_component_latex(field.factor(False), right_weight, color),
                            current=_current_symbol(field),
                        )
                    )
    return terms


def _su3_terms(field: Field) -> list[GaugeInteractionTerm]:
    if field.su3 == SU3Rep(0, 0):
        return []
    terms = []
    matrices = _su3_generators(field.su3)
    colors = _color_labels(field.su3)
    for adjoint_label, matrix in enumerate(matrices, start=1):
        for row, left_color in enumerate(colors):
            for col, right_color in enumerate(colors):
                coeff = sp.simplify(matrix[row, col])
                if coeff == 0:
                    continue
                for weak in _weak_labels(field.su2):
                    terms.append(
                        GaugeInteractionTerm(
                            coefficient=coeff,
                            coupling="g_3",
                            gauge_boson=rf"G^{{{adjoint_label}}}_\mu",
                            left_component=_component_latex(field.factor(True), weak, left_color, conjugate_dual_weight=False),
                            right_component=_component_latex(field.factor(False), weak, right_color),
                            current=_current_symbol(field),
                        )
                    )
    return terms


def _gauge_generator_sets(field: Field) -> list[dict]:
    sets: list[dict] = []
    if field.hypercharge != 0:
        dim = field.su2.dim * len(_color_labels(field.su3))
        sets.append(
            {
                "coupling": "g_1",
                "boson": "B",
                "generators": [(0, sp.eye(dim) * sp.Rational(field.hypercharge.numerator, field.hypercharge.denominator))],
                "labels": _basis_labels(field),
            }
        )
    if field.su2 != SU2Rep(1):
        weak_generators = _su2_generators(field.su2)
        color_dim = len(_color_labels(field.su3))
        sets.append(
            {
                "coupling": "g_2",
                "boson": "W",
                "generators": [(i, sp.kronecker_product(matrix, sp.eye(color_dim))) for i, matrix in enumerate(weak_generators, start=1)],
                "labels": _basis_labels(field),
            }
        )
    if field.su3 != SU3Rep(0, 0):
        color_generators = _su3_generators(field.su3)
        weak_dim = field.su2.dim
        sets.append(
            {
                "coupling": "g_3",
                "boson": "G",
                "generators": [(i, sp.kronecker_product(sp.eye(weak_dim), matrix)) for i, matrix in enumerate(color_generators, start=1)],
                "labels": _basis_labels(field),
            }
        )
    return sets


def _basis_labels(field: Field) -> list[tuple[int, int | tuple[int, int] | None]]:
    return [(weak, color) for weak in _weak_labels(field.su2) for color in _color_labels(field.su3)]


def _matrix_seagull_terms(field: Field, left_set: dict, right_set: dict, left_label, right_label, matrix: sp.Matrix) -> list[SeagullTerm]:
    labels = left_set["labels"]
    terms = []
    for row, left_basis in enumerate(labels):
        for col, right_basis in enumerate(labels):
            coeff = sp.simplify(matrix[row, col])
            if coeff == 0:
                continue
            terms.append(
                SeagullTerm(
                    coefficient=coeff,
                    coupling_left=left_set["coupling"],
                    coupling_right=right_set["coupling"],
                    gauge_left=_gauge_boson_symbol(left_set["boson"], left_label),
                    gauge_right=_gauge_boson_symbol(right_set["boson"], right_label),
                    left_component=_component_latex(
                        field.factor(True),
                        left_basis[0],
                        left_basis[1],
                        conjugate_dual_weight=False,
                    ),
                    right_component=_component_latex(field.factor(False), right_basis[0], right_basis[1]),
                )
            )
    return terms


def _gauge_boson_symbol(prefix: str, label) -> str:
    if prefix == "B":
        return r"B_\mu"
    return rf"{prefix}^{{{label}}}_\mu"


def _su2_f_constants() -> dict[tuple[int, int, int], sp.Expr]:
    constants = {}
    for labels in ((1, 2, 3), (2, 3, 1), (3, 1, 2)):
        constants[labels] = sp.Integer(1)
    for labels in ((2, 1, 3), (3, 2, 1), (1, 3, 2)):
        constants[labels] = sp.Integer(-1)
    return constants


def _nonabelian_self_terms(coupling: str, boson: str, constants: dict[tuple[int, int, int], sp.Expr]) -> list[GaugeSelfInteractionTerm]:
    cubic = []
    quartic = []
    for (a, b, c), coeff in constants.items():
        cubic.append(
            GaugeSelfInteractionTerm(
                -coeff * sp.Symbol(coupling),
                rf"(\partial_\mu {boson}^{{{a}}}_\nu) {boson}^{{{b}\mu}} {boson}^{{{c}\nu}}",
            )
        )
    for (a, b, e), coeff1 in constants.items():
        for (c, d, e2), coeff2 in constants.items():
            if e != e2:
                continue
            quartic.append(
                GaugeSelfInteractionTerm(
                    -sp.Rational(1, 4) * coeff1 * coeff2 * sp.Symbol(coupling) ** 2,
                    rf"{boson}^{{{a}}}_\mu {boson}^{{{b}}}_\nu {boson}^{{{c}\mu}} {boson}^{{{d}\nu}}",
                )
            )
    return cubic + quartic


def _weak_labels(rep: SU2Rep) -> list[int]:
    return _weights(rep)


def _color_labels(rep: SU3Rep) -> list[int | tuple[int, int] | None]:
    return list(_su3_color_labels(rep))


def _su2_generators(rep: SU2Rep) -> tuple[sp.Matrix, sp.Matrix, sp.Matrix]:
    weights = _weak_labels(rep)
    dim = rep.dim
    t_plus = sp.zeros(dim, dim)
    t_minus = sp.zeros(dim, dim)
    t3 = sp.zeros(dim, dim)
    j = sp.Rational(rep.two_j, 2)
    weight_to_index = {weight: index for index, weight in enumerate(weights)}
    for col, two_m in enumerate(weights):
        m = sp.Rational(two_m, 2)
        t3[col, col] = m
        raised = two_m + 2
        lowered = two_m - 2
        if raised in weight_to_index:
            row = weight_to_index[raised]
            t_plus[row, col] = sp.sqrt((j - m) * (j + m + 1))
        if lowered in weight_to_index:
            row = weight_to_index[lowered]
            t_minus[row, col] = sp.sqrt((j + m) * (j - m + 1))
    t1 = (t_plus + t_minus) / 2
    t2 = (t_plus - t_minus) / (2 * sp.I)
    return (t1, t2, t3)


def _su3_generators(rep: SU3Rep) -> tuple[sp.Matrix, ...]:
    return _general_su3_generators(rep)
