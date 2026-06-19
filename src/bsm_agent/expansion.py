"""Component expansion of gauge contractions.

The expansion backend handles SU(2) Clebsch-Gordan factors for scalar
operators and an explicit SU(3) color basis for common low-dimensional scalar
operators.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from itertools import permutations, product
from math import comb, factorial

import sympy as sp
from sympy.physics.wigner import clebsch_gordan

from .fields import FieldFactor, FieldKind, field_latex_name
from .groups import SU2Rep, SU3Rep, su2_symmetric_power
from .operators import latex_identifier
from .scalar_invariants import singlet_basis as _scalar_singlet_basis
from .su3_invariants import singlet_basis as _general_su3_singlet_basis
from .su3_invariants import singlet_multiplicity as _general_su3_singlet_multiplicity
from .su2_invariants import singlet_basis
from .tensor_algebra import SparseTensorObject, contract_tensor_objects

_MAX_EXPANDED_BREAKABLE_TERMS = 2


def _normalize_exact_coeff(expr: sp.Expr) -> sp.Expr:
    """Cheap exact normalization for Clebsch-Gordan coefficients."""

    if expr in {0, 1, -1}:
        return expr
    if not getattr(expr, "is_Add", False):
        return expr
    return sp.cancel(expr)


@dataclass(frozen=True)
class ExpandedTerm:
    coefficient: sp.Expr
    fields: tuple[str, ...]

    def latex(self) -> str:
        body = " ".join(self.fields)
        coeff = self.coefficient
        if coeff == 1:
            return body
        if coeff == -1:
            return f"- {body}"
        return rf"{sp.latex(coeff)}\, {body}"


@dataclass(frozen=True)
class _AssignmentTerm:
    coefficient: sp.Expr
    weak: tuple[int, ...]
    color: tuple[int | str | None, ...]


@dataclass(frozen=True)
class _ReducedFactorGroup:
    factor: FieldFactor
    positions: tuple[int, ...]

    @property
    def count(self) -> int:
        return len(self.positions)


@dataclass(frozen=True)
class _ReducedGroupBasis:
    j3: tuple[tuple[sp.Expr, ...], ...]
    jp: tuple[tuple[sp.Expr, ...], ...]
    jm: tuple[tuple[sp.Expr, ...], ...]
    expansions: tuple[tuple[tuple[sp.Expr, tuple[int, ...]], ...], ...]

    @property
    def dim(self) -> int:
        return len(self.expansions)


def expand_operator(operator) -> list[ExpandedTerm]:
    """Expand a scalar operator into component fields."""

    if operator.category not in {"scalar_potential", "yukawa", "fermion_mass"}:
        raise NotImplementedError("Expanded Clebsch output supports scalar, Yukawa, and fermion mass operators")
    factors = tuple(operator.factors)
    if operator.category == "scalar_potential":
        special_real_y0_terms = _real_y0_special_terms(factors, operator.contraction_index)
        if special_real_y0_terms is not None:
            return special_real_y0_terms
        scalar_terms = _scalar_assignment_terms(factors, operator.contraction_index)
        expanded_components: defaultdict[tuple[tuple[FieldFactor, int, int | str | None], ...], sp.Expr] = defaultdict(
            lambda: sp.Integer(0)
        )
        for term in scalar_terms:
            coefficient = term.coefficient
            for factor, two_m in zip(factors, term.weak):
                coefficient *= _real_y0_component_phase(factor, two_m)
            components = tuple(
                sorted(
                    (
                        (factor, two_m, color_label)
                        for factor, two_m, color_label in zip(factors, term.weak, term.color)
                    ),
                    key=_component_sort_key,
                )
            )
            expanded_components[components] += _normalize_exact_coeff(coefficient)

        return [
            ExpandedTerm(
                coeff,
                tuple(_component_latex(factor, two_m, color_label) for factor, two_m, color_label in components),
            )
            for components, coeff in sorted(expanded_components.items(), key=lambda item: tuple(_component_sort_key(x) for x in item[0]))
            if coeff != 0
        ]

    su2_count = _su2_contraction_count(factors, operator.dual_basis_slots)
    su3_count = _su3_contraction_count(factors)
    if su2_count * su3_count == 0:
        raise ValueError("Operator is not a gauge singlet")

    su2_index = ((operator.contraction_index - 1) % su2_count) + 1
    su3_index = ((operator.contraction_index - 1) // su2_count) + 1
    weak_terms = _su2_assignment_terms(factors, su2_index, operator.dual_basis_slots)
    color_terms = _su3_assignment_terms(factors, su3_index)
    expanded: defaultdict[tuple[str, ...], sp.Expr] = defaultdict(lambda: sp.Integer(0))

    for weak in weak_terms:
        for color in color_terms:
            coefficient = weak.coefficient * color.coefficient
            for factor, two_m in zip(factors, weak.weak):
                coefficient *= _component_phase(factor, two_m, operator.phase_target_fields)
            fields = tuple(
                _component_latex(factor, two_m, color_label, dual_basis=(index in operator.dual_basis_slots))
                for index, (factor, two_m, color_label) in enumerate(zip(factors, weak.weak, color.color))
            )
            expanded[fields] += coefficient

    return [
        ExpandedTerm(coeff, fields)
        for fields, coeff in sorted(expanded.items())
        if coeff != 0
    ]


def _real_y0_quadratic_terms(factors: tuple[FieldFactor, ...]) -> list[ExpandedTerm] | None:
    if len(factors) != 2:
        return None
    left, right = factors
    if left.kind != FieldKind.SCALAR or right.kind != FieldKind.SCALAR:
        return None
    if left.field != right.field:
        return None
    field = left.field
    if not field.real or field.hypercharge != 0:
        return None
    if left.conjugate or right.conjugate:
        return None

    terms: list[ExpandedTerm] = []
    for two_m in _weights(field.su2):
        if two_m < 0:
            continue
        if two_m == 0:
            terms.append(
                ExpandedTerm(
                    sp.Integer(1),
                    (
                        _component_latex(left, 0),
                        _component_latex(right, 0),
                    ),
                )
            )
            continue
        terms.append(
            ExpandedTerm(
                sp.Integer(2),
                (
                    _component_latex(left, -two_m),
                    _component_latex(right, two_m),
                ),
            )
        )
    return terms


def _real_y0_special_terms(factors: tuple[FieldFactor, ...], contraction_index: int) -> list[ExpandedTerm] | None:
    quadratic = _real_y0_quadratic_terms(factors)
    if quadratic is not None:
        return quadratic

    real_field = _single_real_y0_scalar_field(factors)
    if real_field is None:
        return None

    if len(factors) == 4 and all(factor.field == real_field for factor in factors):
        if real_field.su2.dim in {3, 5} and contraction_index == 1:
            return _expanded_poly_to_terms(_poly_mul(_real_y0_norm_poly(real_field), _real_y0_norm_poly(real_field)))
        if real_field.su2.dim == 7:
            if contraction_index == 1:
                return _expanded_poly_to_terms(_poly_mul(_real_y0_norm_poly(real_field), _real_y0_norm_poly(real_field)))
            if contraction_index == 2:
                return _expanded_poly_to_terms(_real_y0_septet_pure03_poly(real_field))
        return None

    if len(factors) == 4:
        generic_blocks = _real_y0_factorized_quartic_terms(factors, contraction_index)
        if generic_blocks is not None:
            return generic_blocks

    return None


def _single_real_y0_scalar_field(factors: tuple[FieldFactor, ...]):
    real_fields = {
        factor.field
        for factor in factors
        if factor.kind == FieldKind.SCALAR and factor.field.real and factor.field.hypercharge == 0
    }
    if len(real_fields) != 1:
        return None
    return next(iter(real_fields))


def _scalar_conjugate_pair(
    factors: tuple[FieldFactor, ...],
    *,
    exclude,
) -> tuple[FieldFactor, FieldFactor] | None:
    by_field: defaultdict = defaultdict(list)
    for factor in factors:
        if factor.kind != FieldKind.SCALAR or factor.field == exclude:
            continue
        by_field[factor.field].append(factor)
    for field, group in by_field.items():
        if len(group) != 2:
            continue
        if sum(factor.conjugate for factor in group) != 1:
            continue
        left = next(factor for factor in group if factor.conjugate)
        right = next(factor for factor in group if not factor.conjugate)
        return left, right
    return None


def _real_y0_factorized_quartic_terms(
    factors: tuple[FieldFactor, ...],
    contraction_index: int,
) -> list[ExpandedTerm] | None:
    if len(factors) != 4 or contraction_index != 1:
        return None

    by_field: defaultdict = defaultdict(list)
    for factor in factors:
        if factor.kind != FieldKind.SCALAR:
            return None
        by_field[factor.field].append(factor)

    blocks: list[dict[tuple[str, ...], sp.Expr]] = []
    for field, group in by_field.items():
        if len(group) != 2:
            return None
        if field.real and field.hypercharge == 0:
            if any(factor.conjugate for factor in group):
                return None
            blocks.append(_real_y0_norm_poly(field))
            continue
        if sum(factor.conjugate for factor in group) != 1:
            return None
        left = next(factor for factor in group if factor.conjugate)
        right = next(factor for factor in group if not factor.conjugate)
        blocks.append(_scalar_pair_poly(left, right))

    if not blocks:
        return None

    poly = blocks[0]
    for block in blocks[1:]:
        poly = _poly_mul(poly, block)
    return _expanded_poly_to_terms(poly)


def _real_y0_norm_poly(field) -> dict[tuple[str, ...], sp.Expr]:
    factor = field.factor(False)
    poly: dict[tuple[str, ...], sp.Expr] = {}
    for two_m in _weights(field.su2):
        if two_m < 0:
            continue
        if two_m == 0:
            fields = tuple(sorted((_component_latex(factor, 0), _component_latex(factor, 0))))
            poly[fields] = sp.Integer(1)
            continue
        fields = tuple(sorted((_component_latex(factor, -two_m), _component_latex(factor, two_m))))
        poly[fields] = sp.Integer(2)
    return poly


def _scalar_pair_poly(left: FieldFactor, right: FieldFactor) -> dict[tuple[str, ...], sp.Expr]:
    poly: dict[tuple[str, ...], sp.Expr] = {}
    for two_m in _weights(left.field.su2):
        fields = tuple(sorted((_component_latex(left, two_m), _component_latex(right, -two_m))))
        poly[fields] = poly.get(fields, sp.Integer(0)) + sp.Integer(1)
    return poly


def _poly_mul(
    left: dict[tuple[str, ...], sp.Expr],
    right: dict[tuple[str, ...], sp.Expr],
) -> dict[tuple[str, ...], sp.Expr]:
    product_poly: dict[tuple[str, ...], sp.Expr] = {}
    for left_fields, left_coeff in left.items():
        for right_fields, right_coeff in right.items():
            fields = tuple(sorted(left_fields + right_fields))
            product_poly[fields] = product_poly.get(fields, sp.Integer(0)) + _normalize_exact_coeff(left_coeff * right_coeff)
    return product_poly


def _expanded_poly_to_terms(poly: dict[tuple[str, ...], sp.Expr]) -> list[ExpandedTerm]:
    return [
        ExpandedTerm(coeff, fields)
        for fields, coeff in sorted(poly.items())
        if coeff != 0
    ]


def _real_y0_septet_pure03_poly(field) -> dict[tuple[str, ...], sp.Expr]:
    factor = field.factor(False)

    def component(charge: Fraction | int) -> str:
        return _component_latex(factor, int(2 * Fraction(charge)))

    phi0 = component(0)
    phip = component(1)
    phin = component(-1)
    phipp = component(2)
    phinn = component(-2)
    phippp = component(3)
    phinnn = component(-3)

    sqrt = sp.sqrt
    return {
        tuple(sorted((phi0, phi0, phi0, phi0))): sp.Rational(11, 25),
        tuple(sorted((phi0, phi0, phin, phip))): sp.Rational(44, 25),
        tuple(sorted((phin, phin, phip, phip))): sp.Rational(142, 75),
        tuple(sorted((phi0, phinn, phip, phip))): sp.Rational(4, 5) * sqrt(sp.Rational(2, 15)),
        tuple(sorted((phinnn, phip, phip, phip))): sp.Rational(8, 5) / sqrt(15),
        tuple(sorted((phi0, phin, phin, phipp))): sp.Rational(4, 5) * sqrt(sp.Rational(2, 15)),
        tuple(sorted((phi0, phi0, phinn, phipp))): sp.Rational(12, 5),
        tuple(sorted((phin, phinn, phip, phipp))): sp.Rational(52, 15),
        tuple(sorted((phi0, phinnn, phip, phipp))): sp.Rational(4, 5) * sqrt(2),
        tuple(sorted((phinn, phinn, phipp, phipp))): sp.Rational(4, 3),
        tuple(sorted((phin, phinnn, phipp, phipp))): -sp.Integer(4) / sqrt(15),
        tuple(sorted((phin, phin, phin, phippp))): sp.Rational(8, 5) / sqrt(15),
        tuple(sorted((phi0, phin, phinn, phippp))): sp.Rational(4, 5) * sqrt(2),
        tuple(sorted((phi0, phi0, phinnn, phippp))): sp.Rational(4, 5),
        tuple(sorted((phinn, phinn, phip, phippp))): -sp.Integer(4) / sqrt(15),
        tuple(sorted((phin, phinnn, phip, phippp))): sp.Rational(12, 5),
        tuple(sorted((phinn, phinnn, phipp, phippp))): sp.Integer(4),
        tuple(sorted((phinnn, phinnn, phippp, phippp))): sp.Integer(2),
    }


def _real_y0_component_phase(factor: FieldFactor, two_m: int) -> sp.Expr:
    if factor.kind != FieldKind.SCALAR or not factor.field.real or factor.field.hypercharge != 0 or factor.conjugate:
        return sp.Integer(1)
    phase_by_dim = {
        3: {2: -1, 0: 1, -2: 1},
        5: {4: 1, 2: sp.I, 0: 1, -2: sp.I, -4: 1},
        7: {6: sp.I, 4: 1, 2: sp.I, 0: 1, -2: sp.I, -4: 1, -6: sp.I},
    }
    return phase_by_dim.get(factor.field.su2.dim, {}).get(two_m, sp.Integer(1))


def expand_operator_su2(operator) -> list[ExpandedTerm]:
    """Expand a scalar operator into SU(2) components.

    The supported basis covers the dominant renormalizable scalar cases:
    two symmetric groups, for example ``H H†`` and ``S S S† S†``.  This is
    enough to expose the independent triplet quartics with explicit CG factors.
    """

    terms = _su2_assignment_terms(tuple(operator.factors), operator.contraction_index)
    expanded = []
    for term in terms:
        fields = tuple(_component_latex(factor, two_m) for factor, two_m in zip(operator.factors, term.weak))
        expanded.append(ExpandedTerm(term.coefficient, fields))
    return expanded


def _breakable_expanded_sum(pieces: list[str]) -> str:
    if not pieces:
        return "0"
    body = pieces[0]
    for piece in pieces[1:]:
        body += r" \allowbreak " + piece
    return body


def expanded_operator_latex(operator) -> str:
    terms = expand_operator(operator)
    if not terms:
        return "0"
    pieces = []
    for index, term in enumerate(terms):
        latex = term.latex()
        if index > 0 and not latex.startswith("-"):
            latex = "+ " + latex
        pieces.append(latex)
    multiline = len(pieces) > _MAX_EXPANDED_BREAKABLE_TERMS
    inner = _breakable_expanded_sum(pieces)
    contraction = rf"^{{({operator.contraction_index})}}" if operator.contraction_count > 1 else ""
    suffix = r" + \mathrm{h.c.}" if operator.add_hc else ""
    if multiline:
        return rf"{latex_identifier(operator.coefficient)}{contraction}\bigl( {inner} \bigr){suffix}"
    return rf"{latex_identifier(operator.coefficient)}{contraction}\left( {inner} \right){suffix}"


def expand_operator_su3(operator) -> list[ExpandedTerm]:
    """Expand supported SU(3) color contractions.

    Supported bases:
    - ``3 x bar3`` via delta
    - ``3 x 3 x 3`` and ``bar3 x bar3 x bar3`` via epsilon
    - ``6 x 6 x 6`` and ``bar6 x bar6 x bar6`` via double epsilon
    - ``3 x bar3 x 3 x bar3`` via the two independent delta pairings
    """

    factors = tuple(operator.factors)
    reps = tuple(factor.su3 for factor in factors)
    fundamental = SU3Rep(1, 0)
    antifundamental = SU3Rep(0, 1)
    sextet = SU3Rep(2, 0)
    antisextet = SU3Rep(0, 2)

    if len(factors) == 2 and sorted(reps) == [antifundamental, fundamental]:
        fund_pos = next(i for i, rep in enumerate(reps) if rep == fundamental)
        anti_pos = next(i for i, rep in enumerate(reps) if rep == antifundamental)
        return _assignments_to_expanded(factors, _su3_delta_assignments(factors, fund_pos, anti_pos))

    if len(factors) == 3 and all(rep == fundamental for rep in reps):
        return _assignments_to_expanded(factors, _su3_epsilon_assignments(factors))

    if len(factors) == 3 and all(rep == antifundamental for rep in reps):
        return _assignments_to_expanded(factors, _su3_epsilon_assignments(factors))

    if len(factors) == 3 and (all(rep == sextet for rep in reps) or all(rep == antisextet for rep in reps)):
        return _assignments_to_expanded(factors, _su3_three_sextet_assignments(factors))

    if len(factors) == 4 and reps.count(fundamental) == 2 and reps.count(antifundamental) == 2:
        pairings = _delta_pairings(reps, fundamental, antifundamental)
        if operator.contraction_index > len(pairings):
            raise ValueError("Contraction index exceeds available SU(3) delta pairings")
        return _assignments_to_expanded(factors, _su3_double_delta_assignments(factors, pairings[operator.contraction_index - 1]))

    if len(factors) == 2 and sorted(reps) == [antisextet, sextet]:
        return _assignments_to_expanded(factors, _su3_sextet_delta_assignments(factors))

    if len(factors) == 3 and reps.count(fundamental) == 2 and reps.count(antisextet) == 1:
        return _assignments_to_expanded(factors, _su3_two_triplet_antisextet_assignments(factors))

    if len(factors) == 3 and reps.count(antifundamental) == 2 and reps.count(sextet) == 1:
        return _assignments_to_expanded(factors, _su3_two_antitriplet_sextet_assignments(factors))

    raise NotImplementedError(f"Explicit SU(3) color expansion is not implemented for reps {[str(rep) for rep in reps]}")


def _su2_contraction_count(factors: tuple[FieldFactor, ...], dual_basis_slots: tuple[int, ...] = ()) -> int:
    return len(_su2_assignment_terms_cached(factors, dual_basis_slots))


def _scalar_contraction_count(factors: tuple[FieldFactor, ...]) -> int:
    return len(_scalar_assignment_terms_cached(factors))


def _su3_contraction_count(factors: tuple[FieldFactor, ...]) -> int:
    if all(factor.su3 == SU3Rep(0, 0) for factor in factors):
        return 1
    non_singlets = tuple(factor for factor in factors if factor.su3 != SU3Rep(0, 0))
    if len(non_singlets) != len(factors):
        return _su3_contraction_count(non_singlets)
    return _general_su3_singlet_multiplicity(non_singlets)


def _su2_assignment_terms(
    factors: tuple[FieldFactor, ...],
    contraction_index: int,
    dual_basis_slots: tuple[int, ...] = (),
) -> list[_AssignmentTerm]:
    basis = _su2_assignment_terms_cached(factors, dual_basis_slots)
    if contraction_index > len(basis):
        raise ValueError("Contraction index exceeds available SU(2) singlets")
    return list(basis[contraction_index - 1])


def _scalar_assignment_terms(factors: tuple[FieldFactor, ...], contraction_index: int) -> list[_AssignmentTerm]:
    basis = _scalar_assignment_terms_cached(factors)
    if contraction_index > len(basis):
        raise ValueError("Contraction index exceeds available scalar singlets")
    return list(basis[contraction_index - 1])


@lru_cache(maxsize=None)
def _scalar_assignment_terms_cached(factors: tuple[FieldFactor, ...]) -> tuple[tuple[_AssignmentTerm, ...], ...]:
    if all(factor.su3 == SU3Rep(0, 0) for factor in factors):
        return _su2_assignment_terms_cached(factors)
    if all(factor.su2 == SU2Rep(1) for factor in factors):
        return _su3_assignment_terms_cached(factors)
    basis = _scalar_singlet_basis(factors)
    return tuple(
        tuple(_AssignmentTerm(term.coefficient, term.weights, term.colors) for term in contraction)
        for contraction in basis
    )


@lru_cache(maxsize=None)
def _su2_assignment_terms_cached(
    factors: tuple[FieldFactor, ...],
    dual_basis_slots: tuple[int, ...] = (),
) -> tuple[tuple[_AssignmentTerm, ...], ...]:
    direct_bilinear_quartics = _direct_bilinear_scalar_quartic_terms(factors)
    if direct_bilinear_quartics:
        return direct_bilinear_quartics
    direct_conjugate_pair_quartics = _direct_conjugate_pair_scalar_quartic_terms(factors)
    if direct_conjugate_pair_quartics:
        return direct_conjugate_pair_quartics
    direct_symmetric_pair_quartics = _direct_symmetric_pair_scalar_quartic_terms(factors)
    if direct_symmetric_pair_quartics:
        return direct_symmetric_pair_quartics
    direct_mixed_pair_quartics = _direct_mixed_pair_scalar_quartic_terms(factors)
    if direct_mixed_pair_quartics:
        return direct_mixed_pair_quartics
    direct_pair_quartics = _direct_pair_channel_scalar_quartic_terms(factors)
    if direct_pair_quartics:
        return direct_pair_quartics
    reduced_groups = _reduced_factor_groups(factors)
    if len(reduced_groups) == 2:
        direct_two_group_terms = _direct_two_group_su2_assignment_terms(reduced_groups, len(factors))
        if direct_two_group_terms:
            return direct_two_group_terms
    reduced_scalar_terms = _reduced_scalar_su2_assignment_terms(factors)
    if reduced_scalar_terms:
        return reduced_scalar_terms
    basis = singlet_basis(_su2_dual_proxy_factors(factors, dual_basis_slots))
    return tuple(
        tuple(_AssignmentTerm(term.coefficient, term.weights, tuple(None for _ in factors)) for term in contraction)
        for contraction in basis
    )


def _su2_dual_proxy_factors(
    factors: tuple[FieldFactor, ...],
    dual_basis_slots: tuple[int, ...],
) -> tuple[FieldFactor, ...]:
    dual_slot_set = set(dual_basis_slots)
    return tuple(
        FieldFactor(factor.field, not factor.conjugate) if index in dual_slot_set else factor
        for index, factor in enumerate(factors)
    )


def _reduced_scalar_su2_assignment_terms(
    factors: tuple[FieldFactor, ...],
) -> tuple[tuple[_AssignmentTerm, ...], ...]:
    if not factors or any(factor.kind != FieldKind.SCALAR for factor in factors):
        return ()
    groups = _reduced_factor_groups(factors)
    group_specs = tuple((group.factor, group.count) for group in groups)
    vectors = _reduced_singlet_vectors(group_specs)
    if not vectors:
        return ()
    bases = tuple(_group_basis(factor, count) for factor, count in group_specs)
    contractions: list[tuple[_AssignmentTerm, ...]] = []
    for vector in vectors:
        expanded = _expand_reduced_vector(vector, groups, bases, len(factors))
        terms = tuple(
            _AssignmentTerm(_normalize_exact_coeff(coeff), weights, tuple(None for _ in range(len(factors))))
            for weights, coeff in sorted(expanded.items())
            if coeff != 0
        )
        if terms:
            contractions.append(terms)
    return tuple(contractions)

def _direct_bilinear_scalar_quartic_terms(factors: tuple[FieldFactor, ...]) -> tuple[tuple[_AssignmentTerm, ...], ...]:
    if len(factors) != 4 or any(factor.kind != FieldKind.SCALAR for factor in factors):
        return ()
    if not _is_alternating_self_quartic_pattern(factors):
        return ()

    left_positions = (0, 1)
    right_positions = (2, 3)
    left_channels = _ordered_pair_channels(factors, left_positions)
    right_channels = _ordered_pair_channels(factors, right_positions)
    common_channels = sorted(set(left_channels).intersection(right_channels), key=lambda rep: rep.dim)
    if not common_channels:
        return ()

    contractions: list[tuple[_AssignmentTerm, ...]] = []
    for channel in common_channels:
        expanded: defaultdict[tuple[int, ...], sp.Expr] = defaultdict(lambda: sp.Integer(0))
        for two_m in _weights(channel):
            singlet_cg = _cg(channel.two_j, two_m, channel.two_j, -two_m, 0, 0)
            if singlet_cg == 0:
                continue
            left_terms = _ordered_pair_assignment_terms(factors, left_positions, channel, two_m)
            right_terms = _ordered_pair_assignment_terms(factors, right_positions, channel, -two_m)
            for left_term in left_terms:
                for right_term in right_terms:
                    weak = [0] * len(factors)
                    for index, value in enumerate(left_term.weak):
                        if value is not None:
                            weak[index] = value
                    for index, value in enumerate(right_term.weak):
                        if value is not None:
                            weak[index] = value
                    expanded[tuple(weak)] += singlet_cg * left_term.coefficient * right_term.coefficient
        terms = tuple(
            _AssignmentTerm(_normalize_exact_coeff(coeff), weights, tuple(None for _ in factors))
            for weights, coeff in sorted(expanded.items())
            if coeff != 0
        )
        if terms:
            contractions.append(terms)
    return tuple(contractions)


def _direct_conjugate_pair_scalar_quartic_terms(factors: tuple[FieldFactor, ...]) -> tuple[tuple[_AssignmentTerm, ...], ...]:
    if len(factors) != 4 or any(factor.kind != FieldKind.SCALAR for factor in factors):
        return ()

    grouped_positions: defaultdict = defaultdict(dict)
    for index, factor in enumerate(factors):
        positions = grouped_positions[factor.field]
        if factor.conjugate in positions:
            return ()
        positions[factor.conjugate] = index

    if len(grouped_positions) != 2:
        return ()

    pair_positions: list[tuple[int, int]] = []
    for field in sorted(grouped_positions, key=lambda item: item.name):
        positions = grouped_positions[field]
        if set(positions) != {False, True}:
            return ()
        pair_positions.append((positions[False], positions[True]))

    left_positions, right_positions = pair_positions
    left_channels = _ordered_pair_channels(factors, left_positions)
    right_channels = _ordered_pair_channels(factors, right_positions)
    common_channels = sorted(set(left_channels).intersection(right_channels), key=lambda rep: rep.dim)
    if not common_channels:
        return ()

    contractions: list[tuple[_AssignmentTerm, ...]] = []
    for channel in common_channels:
        expanded: defaultdict[tuple[int, ...], sp.Expr] = defaultdict(lambda: sp.Integer(0))
        for two_m in _weights(channel):
            singlet_cg = _cg(channel.two_j, two_m, channel.two_j, -two_m, 0, 0)
            if singlet_cg == 0:
                continue
            left_terms = _ordered_pair_assignment_terms(factors, left_positions, channel, two_m)
            right_terms = _ordered_pair_assignment_terms(factors, right_positions, channel, -two_m)
            for left_term in left_terms:
                for right_term in right_terms:
                    weak = [0] * len(factors)
                    for index, value in enumerate(left_term.weak):
                        if value is not None:
                            weak[index] = value
                    for index, value in enumerate(right_term.weak):
                        if value is not None:
                            weak[index] = value
                    expanded[tuple(weak)] += singlet_cg * left_term.coefficient * right_term.coefficient
        terms = tuple(
            _AssignmentTerm(_normalize_exact_coeff(coeff), weights, tuple(None for _ in factors))
            for weights, coeff in sorted(expanded.items())
            if coeff != 0
        )
        if terms:
            contractions.append(terms)
    return tuple(contractions)



def _direct_symmetric_pair_scalar_quartic_terms(factors: tuple[FieldFactor, ...]) -> tuple[tuple[_AssignmentTerm, ...], ...]:
    if len(factors) != 4 or any(factor.kind != FieldKind.SCALAR for factor in factors):
        return ()

    groups = _factor_index_groups(factors)
    if len(groups) != 3:
        return ()

    identical_groups = [group for group in groups if group.count == 2]
    single_groups = [group for group in groups if group.count == 1]
    if len(identical_groups) != 1 or len(single_groups) != 2:
        return ()

    identical = identical_groups[0]
    ordered_pair_positions = tuple(sorted((single_groups[0].positions[0], single_groups[1].positions[0])))

    if set(identical.positions).intersection(ordered_pair_positions):
        return ()
    if set(identical.positions).union(ordered_pair_positions) != set(range(len(factors))):
        return ()

    identical_channels = _symmetric_channels(identical.factor.su2, identical.count)
    pair_channels = set(_ordered_pair_channels(factors, ordered_pair_positions))
    common_channels = sorted(identical_channels.intersection(pair_channels), key=lambda rep: rep.dim)
    if not common_channels:
        return ()

    contractions: list[tuple[_AssignmentTerm, ...]] = []
    for channel in common_channels:
        expanded: defaultdict[tuple[int, ...], sp.Expr] = defaultdict(lambda: sp.Integer(0))
        for two_m in _weights(channel):
            singlet_cg = _cg(channel.two_j, two_m, channel.two_j, -two_m, 0, 0)
            if singlet_cg == 0:
                continue
            identical_terms = _state_assignment_terms(identical, channel, two_m, len(factors))
            pair_terms = _ordered_pair_assignment_terms(factors, ordered_pair_positions, channel, -two_m)
            for left_term in identical_terms:
                for right_term in pair_terms:
                    weak = [0] * len(factors)
                    for index, value in enumerate(left_term.weak):
                        if value is not None:
                            weak[index] = value
                    for index, value in enumerate(right_term.weak):
                        if value is not None:
                            weak[index] = value
                    expanded[tuple(weak)] += singlet_cg * left_term.coefficient * right_term.coefficient
        terms = tuple(
            _AssignmentTerm(_normalize_exact_coeff(coeff), weights, tuple(None for _ in factors))
            for weights, coeff in sorted(expanded.items())
            if coeff != 0
        )
        if terms:
            contractions.append(terms)
    return tuple(contractions)


def _direct_mixed_pair_scalar_quartic_terms(factors: tuple[FieldFactor, ...]) -> tuple[tuple[_AssignmentTerm, ...], ...]:
    if len(factors) != 4 or any(factor.kind != FieldKind.SCALAR for factor in factors):
        return ()

    groups = _factor_index_groups(factors)
    if len(groups) != 3:
        return ()

    identical_groups = [group for group in groups if group.count == 2]
    single_groups = [group for group in groups if group.count == 1]
    if len(identical_groups) != 1 or len(single_groups) != 2:
        return ()

    identical = identical_groups[0]
    first, second = single_groups
    if first.factor.field != second.factor.field or first.factor.conjugate == second.factor.conjugate:
        return ()

    conjugate_pair_positions = (first.positions[0], second.positions[0])
    if first.factor.conjugate:
        conjugate_pair_positions = (second.positions[0], first.positions[0])

    identical_positions = identical.positions
    if set(identical_positions).intersection(conjugate_pair_positions):
        return ()
    if set(identical_positions).union(conjugate_pair_positions) != set(range(len(factors))):
        return ()

    identical_channels = _symmetric_channels(identical.factor.su2, identical.count)
    conjugate_channels = set(_ordered_pair_channels(factors, conjugate_pair_positions))
    common_channels = sorted(identical_channels.intersection(conjugate_channels), key=lambda rep: rep.dim)
    if not common_channels:
        return ()

    contractions: list[tuple[_AssignmentTerm, ...]] = []
    for channel in common_channels:
        expanded: defaultdict[tuple[int, ...], sp.Expr] = defaultdict(lambda: sp.Integer(0))
        for two_m in _weights(channel):
            singlet_cg = _cg(channel.two_j, two_m, channel.two_j, -two_m, 0, 0)
            if singlet_cg == 0:
                continue
            identical_terms = _state_assignment_terms(identical, channel, two_m, len(factors))
            conjugate_terms = _ordered_pair_assignment_terms(factors, conjugate_pair_positions, channel, -two_m)
            for left_term in identical_terms:
                for right_term in conjugate_terms:
                    weak = [0] * len(factors)
                    for index, value in enumerate(left_term.weak):
                        if value is not None:
                            weak[index] = value
                    for index, value in enumerate(right_term.weak):
                        if value is not None:
                            weak[index] = value
                    expanded[tuple(weak)] += singlet_cg * left_term.coefficient * right_term.coefficient
        terms = tuple(
            _AssignmentTerm(_normalize_exact_coeff(coeff), weights, tuple(None for _ in factors))
            for weights, coeff in sorted(expanded.items())
            if coeff != 0
        )
        if terms:
            contractions.append(terms)
    return tuple(contractions)


def _direct_pair_channel_scalar_quartic_terms(factors: tuple[FieldFactor, ...]) -> tuple[tuple[_AssignmentTerm, ...], ...]:
    if len(factors) != 4 or any(factor.kind != FieldKind.SCALAR for factor in factors):
        return ()
    self_quartics = _direct_self_scalar_quartic_terms(factors)
    if self_quartics:
        return self_quartics
    groups = _factor_index_groups(factors)
    if len(groups) != 2 or any(group.count != 2 for group in groups):
        return ()

    left, right = groups
    common_channels = sorted(
        set(_symmetric_channels(left.factor.su2, left.count)).intersection(_symmetric_channels(right.factor.su2, right.count)),
        key=lambda rep: rep.dim,
    )
    if not common_channels:
        return ()

    contractions: list[tuple[_AssignmentTerm, ...]] = []
    for channel in common_channels:
        expanded: defaultdict[tuple[int, ...], sp.Expr] = defaultdict(lambda: sp.Integer(0))
        for two_m in _weights(channel):
            singlet_cg = _cg(channel.two_j, two_m, channel.two_j, -two_m, 0, 0)
            if singlet_cg == 0:
                continue
            left_terms = _state_assignment_terms(left, channel, two_m, len(factors))
            right_terms = _state_assignment_terms(right, channel, -two_m, len(factors))
            for left_term in left_terms:
                for right_term in right_terms:
                    weak = [0] * len(factors)
                    for index, value in enumerate(left_term.weak):
                        if value is not None:
                            weak[index] = value
                    for index, value in enumerate(right_term.weak):
                        if value is not None:
                            weak[index] = value
                    expanded[tuple(weak)] += singlet_cg * left_term.coefficient * right_term.coefficient
        terms = tuple(
            _AssignmentTerm(_normalize_exact_coeff(coeff), weights, tuple(None for _ in factors))
            for weights, coeff in sorted(expanded.items())
            if coeff != 0
        )
        if terms:
            contractions.append(terms)
    return tuple(contractions)


def _direct_self_scalar_quartic_terms(factors: tuple[FieldFactor, ...]) -> tuple[tuple[_AssignmentTerm, ...], ...]:
    groups = _factor_index_groups(factors)
    if len(groups) != 2 or any(group.count != 2 for group in groups):
        return ()
    left, right = groups
    if left.factor.field != right.factor.field or left.factor.conjugate == right.factor.conjugate:
        return ()
    if left.factor.conjugate:
        left, right = right, left

    rep = left.factor.su2
    degree = rep.dim - 1
    contractions: list[tuple[_AssignmentTerm, ...]] = []
    binom = [comb(degree, twos) for twos in range(degree + 1)]
    for crossed in range(degree // 2 + 1):
        expanded: defaultdict[tuple[int, ...], sp.Expr] = defaultdict(lambda: sp.Integer(0))
        suffix = degree - crossed
        prefix_binom = [comb(crossed, twos) for twos in range(crossed + 1)]
        suffix_binom = [comb(suffix, twos) for twos in range(suffix + 1)]
        for left_prefix_twos in range(crossed + 1):
            left_prefix_mult = prefix_binom[left_prefix_twos]
            for right_prefix_twos in range(crossed + 1):
                prefix_mult = left_prefix_mult * prefix_binom[right_prefix_twos]
                for left_suffix_twos in range(suffix + 1):
                    left_twos = left_prefix_twos + left_suffix_twos
                    left_weight = degree - 2 * left_twos
                    left_norm = sp.sqrt(binom[left_twos])
                    left_mult = prefix_mult * suffix_binom[left_suffix_twos]
                    for right_suffix_twos in range(suffix + 1):
                        right_twos = right_prefix_twos + right_suffix_twos
                        right_weight = degree - 2 * right_twos
                        conj_a_twos = left_prefix_twos + right_suffix_twos
                        conj_b_twos = right_prefix_twos + left_suffix_twos
                        conj_a_weight = 2 * conj_a_twos - degree
                        conj_b_weight = 2 * conj_b_twos - degree
                        multiplicity = left_mult * suffix_binom[right_suffix_twos]
                        coeff = sp.Integer(multiplicity) / (
                            left_norm
                            * sp.sqrt(binom[right_twos])
                            * sp.sqrt(binom[conj_a_twos])
                            * sp.sqrt(binom[conj_b_twos])
                        )
                        weights = [0] * len(factors)
                        weights[left.positions[0]] = left_weight
                        weights[left.positions[1]] = right_weight
                        weights[right.positions[0]] = conj_a_weight
                        weights[right.positions[1]] = conj_b_weight
                        expanded[tuple(weights)] += coeff
        terms = tuple(
            _AssignmentTerm(_normalize_exact_coeff(coeff), weights, tuple(None for _ in factors))
            for weights, coeff in sorted(expanded.items())
            if coeff != 0
        )
        if terms:
            contractions.append(terms)
    return tuple(contractions)


def _is_alternating_self_quartic_pattern(factors: tuple[FieldFactor, ...]) -> bool:
    if len(factors) != 4:
        return False
    if any(factor.kind != FieldKind.SCALAR for factor in factors):
        return False
    names = {factor.field.name for factor in factors}
    if len(names) != 1:
        return False
    reps = {factor.su2 for factor in factors}
    if len(reps) != 1:
        return False
    pattern = tuple(factor.conjugate for factor in factors)
    return pattern in {(False, True, False, True), (True, False, True, False)}


def _ordered_pair_channels(
    factors: tuple[FieldFactor, ...],
    positions: tuple[int, int],
) -> tuple[SU2Rep, ...]:
    left = factors[positions[0]]
    right = factors[positions[1]]
    return tuple(sorted(_su2_pair_products(left.su2, right.su2), key=lambda rep: rep.dim))


def _ordered_pair_assignment_terms(
    factors: tuple[FieldFactor, ...],
    positions: tuple[int, int],
    channel: SU2Rep,
    two_m: int,
) -> tuple[_AssignmentTerm, ...]:
    return _ordered_pair_assignment_terms_cached(factors, positions, channel, two_m)


@lru_cache(maxsize=None)
def _ordered_pair_assignment_terms_cached(
    factors: tuple[FieldFactor, ...],
    positions: tuple[int, int],
    channel: SU2Rep,
    two_m: int,
) -> tuple[_AssignmentTerm, ...]:
    first = factors[positions[0]]
    second = factors[positions[1]]
    expanded: list[_AssignmentTerm] = []
    first_weights = _basis_weights(first.su2, dual=first.conjugate)
    second_weights = _basis_weights(second.su2, dual=second.conjugate)
    empty_colors = tuple(None for _ in range(len(factors)))
    for two_m1 in first_weights:
        for two_m2 in second_weights:
            if two_m1 + two_m2 != two_m:
                continue
            coeff = _cg(first.su2.two_j, two_m1, second.su2.two_j, two_m2, channel.two_j, two_m)
            if coeff == 0:
                continue
            weak = [None] * len(factors)
            weak[positions[0]] = two_m1
            weak[positions[1]] = two_m2
            expanded.append(_AssignmentTerm(coeff, tuple(weak), empty_colors))
    return tuple(expanded)

def _reduced_factor_groups(factors: tuple[FieldFactor, ...]) -> tuple[_ReducedFactorGroup, ...]:
    grouped: list[_ReducedFactorGroup] = []
    seen: set[int] = set()
    for index, factor in enumerate(factors):
        if index in seen:
            continue
        if factor.kind == FieldKind.SCALAR:
            positions = tuple(i for i, other in enumerate(factors) if other == factor)
            for pos in positions:
                seen.add(pos)
            grouped.append(_ReducedFactorGroup(factor, positions))
            continue
        seen.add(index)
        grouped.append(_ReducedFactorGroup(factor, (index,)))
    return tuple(grouped)


def _direct_two_group_su2_assignment_terms(
    groups: tuple[_ReducedFactorGroup, ...],
    total_length: int,
) -> tuple[tuple[_AssignmentTerm, ...], ...]:
    left, right = groups
    # For simple bilinears like H H†, the exact generic singlet basis is cheap
    # and handles the dual-basis convention correctly. The reduced two-group
    # shortcut was introduced for larger symmetric products and gives the wrong
    # relative sign in this singleton-singleton case.
    if left.count == 1 and right.count == 1:
        return ()
    left_states = _group_irrep_states(left.factor, left.count)
    right_states = _group_irrep_states(right.factor, right.count)
    common_channels = sorted(set(left_states).intersection(right_states), key=lambda rep: rep.dim)
    if not common_channels:
        return ()

    contractions: list[tuple[_AssignmentTerm, ...]] = []
    for channel in common_channels:
        left_copies = left_states[channel]
        right_copies = right_states[channel]
        for left_copy in left_copies:
            for right_copy in right_copies:
                expanded: defaultdict[tuple[int, ...], sp.Expr] = defaultdict(lambda: sp.Integer(0))
                for two_m in _weights(channel):
                    cg = _cg(channel.two_j, two_m, channel.two_j, -two_m, 0, 0)
                    if cg == 0:
                        continue
                    for left_coeff, left_weights in left_copy[two_m]:
                        for right_coeff, right_weights in right_copy[-two_m]:
                            weights = [0] * total_length
                            for pos, weight in zip(left.positions, left_weights):
                                weights[pos] = weight
                            for pos, weight in zip(right.positions, right_weights):
                                weights[pos] = weight
                            expanded[tuple(weights)] += cg * left_coeff * right_coeff
                terms = tuple(
                    _AssignmentTerm(_normalize_exact_coeff(coeff), weights, tuple(None for _ in range(total_length)))
                    for weights, coeff in sorted(expanded.items())
                    if coeff != 0
                )
                if terms:
                    contractions.append(terms)
    return tuple(contractions)

@lru_cache(maxsize=None)
def _group_basis(factor: FieldFactor, count: int) -> _ReducedGroupBasis:
    if count == 1:
        return _single_factor_group_basis(factor)
    if factor.kind != FieldKind.SCALAR:
        raise NotImplementedError("Only identical scalar factors can be reduced in the symmetric SU(2) basis")
    return _symmetric_scalar_group_basis(factor, count)


def _single_factor_group_basis(factor: FieldFactor) -> _ReducedGroupBasis:
    rep = factor.su2
    dual = factor.conjugate
    weights = _basis_weights(rep, dual=dual)
    j3, jp, jm = _rep_generators(rep, dual=dual)
    expansions = tuple(((sp.Integer(1), (weight,)),) for weight in weights)
    return _ReducedGroupBasis(_matrix_to_tuple(j3), _matrix_to_tuple(jp), _matrix_to_tuple(jm), expansions)


def _symmetric_scalar_group_basis(factor: FieldFactor, count: int) -> _ReducedGroupBasis:
    rep = factor.su2
    dual = factor.conjugate
    single_j3, single_jp, single_jm = _rep_generators(rep, dual=dual)
    basis_weights = _basis_weights(rep, dual=dual)
    occupations = _occupation_basis(rep.dim, count)
    index = {occupation: i for i, occupation in enumerate(occupations)}
    dim = len(occupations)
    j3 = sp.zeros(dim)
    jp = sp.zeros(dim)
    jm = sp.zeros(dim)

    for column, occupation in enumerate(occupations):
        j3[column, column] = sum(mult * sp.Rational(weight, 2) for mult, weight in zip(occupation, basis_weights))
        for source in range(rep.dim):
            source_mult = occupation[source]
            if source_mult == 0:
                continue
            if source > 0:
                target = source - 1
                coeff = single_jp[target, source]
                if coeff != 0:
                    updated = list(occupation)
                    updated[source] -= 1
                    updated[target] += 1
                    row = index[tuple(updated)]
                    jp[row, column] += sp.sqrt(source_mult * updated[target]) * coeff
            if source < rep.dim - 1:
                target = source + 1
                coeff = single_jm[target, source]
                if coeff != 0:
                    updated = list(occupation)
                    updated[source] -= 1
                    updated[target] += 1
                    row = index[tuple(updated)]
                    jm[row, column] += sp.sqrt(source_mult * updated[target]) * coeff

    expansions = tuple(_occupation_expansion(occupation, basis_weights) for occupation in occupations)
    return _ReducedGroupBasis(_matrix_to_tuple(j3), _matrix_to_tuple(jp), _matrix_to_tuple(jm), expansions)


@lru_cache(maxsize=None)
def _group_irrep_states(
    factor: FieldFactor,
    count: int,
) -> dict[SU2Rep, tuple[dict[int, tuple[tuple[sp.Expr, tuple[int, ...]], ...]], ...]]:
    basis = _group_basis(factor, count)
    if count == 1:
        states = {}
        channel = factor.su2
        copy = {two_m: ((sp.Integer(1), (two_m,)),) for two_m in _basis_weights(channel, dual=factor.conjugate)}
        states[channel] = (copy,)
        return states

    j3 = sp.Matrix(basis.j3)
    jp = sp.Matrix(basis.jp)
    jm = sp.Matrix(basis.jm)
    weight_blocks = _group_weight_blocks(basis)
    result: dict[SU2Rep, list[dict[int, tuple[tuple[sp.Expr, tuple[int, ...]], ...]]]] = defaultdict(list)

    for two_m in sorted((weight for weight in weight_blocks if weight >= 0), reverse=True):
        columns = weight_blocks[two_m]
        if not columns:
            continue
        highest_matrix = jp.extract(range(jp.rows), columns)
        highest_vectors = _orthonormalize_vectors(
            _canonicalize_vector(_embed_subspace_vector(vector, columns, basis.dim))
            for vector in highest_matrix.nullspace()
        )
        channel = SU2Rep(two_m + 1)
        for highest in highest_vectors:
            copy_states: dict[int, tuple[tuple[sp.Expr, tuple[int, ...]], ...]] = {}
            current = highest
            current_two_m = two_m
            while current_two_m >= -two_m:
                copy_states[current_two_m] = _state_vector_expansion(current, basis.expansions)
                if current_two_m == -two_m:
                    break
                lowered = jm * sp.Matrix(current)
                current = _normalize_state_vector(
                    tuple(lowered[row, 0] for row in range(lowered.rows)),
                    channel.two_j,
                    current_two_m,
                )
                current_two_m -= 2
            result[channel].append(copy_states)
    return {rep: tuple(copies) for rep, copies in result.items()}


@lru_cache(maxsize=None)
def _reduced_singlet_vectors(
    group_specs: tuple[tuple[FieldFactor, int], ...],
) -> tuple[tuple[sp.Expr, ...], ...]:
    bases = tuple(_group_basis(factor, count) for factor, count in group_specs)
    dims = [basis.dim for basis in bases]
    size = 1
    for dim in dims:
        size *= dim

    total_jp = sp.zeros(size)
    for position, basis in enumerate(bases):
        jp = sp.Matrix(basis.jp)
        left_dim = 1
        right_dim = 1
        for dim in dims[:position]:
            left_dim *= dim
        for dim in dims[position + 1 :]:
            right_dim *= dim
        total_jp += sp.kronecker_product(sp.eye(left_dim), jp, sp.eye(right_dim))

    zero_indices = _zero_weight_indices(bases)
    if not zero_indices:
        return ()
    system = total_jp.extract(range(total_jp.rows), zero_indices)
    vectors = []
    for reduced_vector in system.nullspace():
        full_vector = [sp.Integer(0)] * size
        for column, basis_index in enumerate(zero_indices):
            full_vector[basis_index] = reduced_vector[column]
        vectors.append(tuple(full_vector))
    return tuple(_deduplicate_vectors(_canonicalize_vector(vector) for vector in vectors))


def _expand_reduced_vector(
    vector: tuple[sp.Expr, ...],
    groups: tuple[_ReducedFactorGroup, ...],
    bases: tuple[_ReducedGroupBasis, ...],
    total_length: int,
) -> defaultdict[tuple[int, ...], sp.Expr]:
    dims = [basis.dim for basis in bases]
    expanded: defaultdict[tuple[int, ...], sp.Expr] = defaultdict(lambda: sp.Integer(0))
    for flat_index, vector_coeff in enumerate(vector):
        if vector_coeff == 0:
            continue
        positions = _flat_index_to_positions(flat_index, dims)
        group_terms = [bases[group_index].expansions[position] for group_index, position in enumerate(positions)]
        for pieces in product(*group_terms):
            weights = [0] * total_length
            coeff = vector_coeff
            for group, piece in zip(groups, pieces):
                piece_coeff, piece_weights = piece
                coeff *= piece_coeff
                for pos, weight in zip(group.positions, piece_weights):
                    weights[pos] = weight
            expanded[tuple(weights)] += coeff
    return expanded


def _occupation_basis(dim: int, count: int) -> tuple[tuple[int, ...], ...]:
    states: list[tuple[int, ...]] = []

    def walk(slot: int, remaining: int, current: list[int]) -> None:
        if slot == dim - 1:
            states.append(tuple(current + [remaining]))
            return
        for used in range(remaining, -1, -1):
            walk(slot + 1, remaining - used, current + [used])

    walk(0, count, [])
    return tuple(states)


def _occupation_expansion(
    occupation: tuple[int, ...],
    basis_weights: tuple[int, ...],
) -> tuple[tuple[sp.Expr, tuple[int, ...]], ...]:
    entries: list[int] = []
    for slot, multiplicity in enumerate(occupation):
        entries.extend([basis_weights[slot]] * multiplicity)
    ordered = tuple(entries)
    states = sorted(set(permutations(ordered)))
    multiplicity = factorial(len(entries))
    for value in occupation:
        multiplicity //= factorial(value)
    coeff = sp.Integer(1) / sp.sqrt(multiplicity)
    return tuple((coeff, tuple(state)) for state in states)


def _group_weight_blocks(basis: _ReducedGroupBasis) -> dict[int, list[int]]:
    blocks: dict[int, list[int]] = defaultdict(list)
    for index in range(basis.dim):
        two_m = int(2 * sp.Rational(basis.j3[index][index]))
        blocks[two_m].append(index)
    return dict(blocks)


def _embed_subspace_vector(
    vector: sp.Matrix | tuple[sp.Expr, ...],
    columns: list[int],
    dim: int,
) -> tuple[sp.Expr, ...]:
    full = [sp.Integer(0)] * dim
    for local_index, column in enumerate(columns):
        full[column] = _normalize_exact_coeff(vector[local_index])
    return tuple(full)


def _orthonormalize_vectors(vectors) -> list[tuple[sp.Expr, ...]]:
    basis: list[tuple[sp.Expr, ...]] = []
    for vector in vectors:
        current = tuple(vector)
        for existing in basis:
            overlap = _state_inner_product(existing, current)
            if overlap != 0:
                current = tuple(_normalize_exact_coeff(entry - overlap * ref) for entry, ref in zip(current, existing))
        if all(entry == 0 for entry in current):
            continue
        basis.append(_normalize_vector(current))
    return basis


def _state_inner_product(left: tuple[sp.Expr, ...], right: tuple[sp.Expr, ...]) -> sp.Expr:
    return _normalize_exact_coeff(sum(sp.conjugate(a) * b for a, b in zip(left, right)))


def _normalize_vector(vector: tuple[sp.Expr, ...]) -> tuple[sp.Expr, ...]:
    norm_sq = _state_inner_product(vector, vector)
    if norm_sq == 0:
        return tuple(sp.Integer(0) for _ in vector)
    norm = sp.sqrt(_normalize_exact_coeff(norm_sq))
    return _canonicalize_vector(tuple(_normalize_exact_coeff(entry / norm) for entry in vector))


def _normalize_state_vector(vector: tuple[sp.Expr, ...], two_j: int, two_m: int) -> tuple[sp.Expr, ...]:
    j = sp.Rational(two_j, 2)
    m = sp.Rational(two_m, 2)
    factor = sp.sqrt((j + m) * (j - m + 1))
    return _normalize_vector(tuple(_normalize_exact_coeff(entry / factor) for entry in vector))


def _state_vector_expansion(
    vector: tuple[sp.Expr, ...],
    basis_expansions: tuple[tuple[tuple[sp.Expr, tuple[int, ...]], ...], ...],
) -> tuple[tuple[sp.Expr, tuple[int, ...]], ...]:
    expanded: defaultdict[tuple[int, ...], sp.Expr] = defaultdict(lambda: sp.Integer(0))
    for coeff, pieces in zip(vector, basis_expansions):
        if coeff == 0:
            continue
        for piece_coeff, weights in pieces:
            expanded[weights] += coeff * piece_coeff
    return tuple((_normalize_exact_coeff(coeff), weights) for weights, coeff in sorted(expanded.items()) if coeff != 0)


def _matrix_to_tuple(matrix: sp.Matrix) -> tuple[tuple[sp.Expr, ...], ...]:
    return tuple(tuple(_normalize_exact_coeff(matrix[row, column]) for column in range(matrix.cols)) for row in range(matrix.rows))


def _flat_index_to_positions(index: int, dims: list[int]) -> tuple[int, ...]:
    positions = [0] * len(dims)
    remainder = index
    for slot in range(len(dims) - 1, -1, -1):
        positions[slot] = remainder % dims[slot]
        remainder //= dims[slot]
    return tuple(positions)


def _zero_weight_indices(bases: tuple[_ReducedGroupBasis, ...]) -> list[int]:
    group_weights = [tuple(_normalize_exact_coeff(basis.j3[index][index]) for index in range(basis.dim)) for basis in bases]
    indices: list[int] = []

    def walk(group_index: int, total_weight: sp.Expr, flat_index: int) -> None:
        if group_index == len(bases):
            if _normalize_exact_coeff(total_weight) == 0:
                indices.append(flat_index)
            return
        basis = bases[group_index]
        for state_index, weight in enumerate(group_weights[group_index]):
            walk(group_index + 1, _normalize_exact_coeff(total_weight + weight), flat_index * basis.dim + state_index)

    walk(0, sp.Integer(0), 0)
    return indices


def _su3_assignment_terms(factors: tuple[FieldFactor, ...], contraction_index: int) -> list[_AssignmentTerm]:
    basis = _su3_assignment_terms_cached(factors)
    if contraction_index > len(basis):
        raise ValueError("Contraction index exceeds available SU(3) singlets")
    return list(basis[contraction_index - 1])


@lru_cache(maxsize=None)
def _su3_assignment_terms_cached(factors: tuple[FieldFactor, ...]) -> tuple[tuple[_AssignmentTerm, ...], ...]:
    if all(factor.su3 == SU3Rep(0, 0) for factor in factors):
        return ((
            _AssignmentTerm(sp.Integer(1), tuple(0 for _ in factors), tuple(None for _ in factors)),
        ),)

    non_singlet_positions = [index for index, factor in enumerate(factors) if factor.su3 != SU3Rep(0, 0)]
    if len(non_singlet_positions) != len(factors):
        sub_factors = tuple(factors[index] for index in non_singlet_positions)
        sub_basis = _su3_assignment_terms_cached(sub_factors)
        embedded_basis = []
        for sub_terms in sub_basis:
            embedded = []
            for term in sub_terms:
                colors = [None] * len(factors)
                for sub_index, original_index in enumerate(non_singlet_positions):
                    colors[original_index] = term.color[sub_index]
                embedded.append(_AssignmentTerm(term.coefficient, tuple(0 for _ in factors), tuple(colors)))
            embedded_basis.append(tuple(embedded))
        return tuple(embedded_basis)

    basis = _general_su3_singlet_basis(factors)
    return tuple(
        tuple(_AssignmentTerm(term.coefficient, tuple(0 for _ in factors), term.colors) for term in contraction)
        for contraction in basis
    )


def _su2_recoupling_paths(reps: tuple[SU2Rep, ...]) -> list[tuple[SU2Rep, ...]]:
    if not reps:
        return []
    if len(reps) == 1:
        return [()] if reps[0] == SU2Rep(1) else []

    paths: list[tuple[SU2Rep, ...]] = []

    def walk(current: SU2Rep, index: int, intermediates: tuple[SU2Rep, ...]) -> None:
        if index == len(reps):
            if current == SU2Rep(1):
                paths.append(intermediates)
            return
        for out_rep in sorted(_su2_pair_products(current, reps[index]), key=lambda rep: rep.dim):
            walk(out_rep, index + 1, intermediates + (out_rep,))

    walk(reps[0], 1, ())
    return paths


def _su2_pair_products(left: SU2Rep, right: SU2Rep) -> set[SU2Rep]:
    lo = abs(left.two_j - right.two_j)
    hi = left.two_j + right.two_j
    return {SU2Rep(two_j + 1) for two_j in range(lo, hi + 1, 2)}


def _su2_recoupling_assignment_terms(factors: tuple[FieldFactor, ...], contraction_index: int) -> list[_AssignmentTerm]:
    reps = tuple(factor.su2 for factor in factors)
    paths = _su2_recoupling_paths(reps)
    if contraction_index > len(paths):
        raise ValueError("Contraction index exceeds available SU(2) recoupling paths")
    path = paths[contraction_index - 1]
    expanded: defaultdict[tuple[int, ...], sp.Expr] = defaultdict(lambda: sp.Integer(0))

    def walk(index: int, current_rep: SU2Rep, current_m: int, coeff: sp.Expr, weights: tuple[int, ...]) -> None:
        if index == len(factors):
            if current_rep == SU2Rep(1) and current_m == 0:
                expanded[weights] += coeff
            return
        out_rep = path[index - 1]
        for next_m in _weights(reps[index]):
            out_m = current_m + next_m
            if out_m < -out_rep.two_j or out_m > out_rep.two_j or (out_m + out_rep.two_j) % 2:
                continue
            cg = _cg(current_rep.two_j, current_m, reps[index].two_j, next_m, out_rep.two_j, out_m)
            if cg != 0:
                walk(index + 1, out_rep, out_m, coeff * cg, weights + (next_m,))

    for first_m in _weights(reps[0]):
        walk(1, reps[0], first_m, sp.Integer(1), (first_m,))

    return [
        _AssignmentTerm(_normalize_exact_coeff(coeff), weights, tuple(None for _ in factors))
        for weights, coeff in sorted(expanded.items())
        if sp.simplify(coeff) != 0
    ]


def _assignments_to_expanded(factors: tuple[FieldFactor, ...], assignments: list[_AssignmentTerm]) -> list[ExpandedTerm]:
    return [
        ExpandedTerm(
            assignment.coefficient,
            tuple(_component_latex(factor, 0, color) for factor, color in zip(factors, assignment.color)),
        )
        for assignment in assignments
    ]


def _su3_delta_assignments(factors: tuple[FieldFactor, ...], fund_pos: int, anti_pos: int) -> list[_AssignmentTerm]:
    terms = []
    for color in range(3):
        colored = [None] * len(factors)
        colored[fund_pos] = color
        colored[anti_pos] = color
        terms.append(_AssignmentTerm(sp.Integer(1), tuple(0 for _ in factors), tuple(colored)))
    return terms


def _su3_double_delta_assignments(
    factors: tuple[FieldFactor, ...],
    pairing: tuple[tuple[int, int], tuple[int, int]],
) -> list[_AssignmentTerm]:
    expanded: defaultdict[tuple[int | None, ...], sp.Expr] = defaultdict(lambda: sp.Integer(0))
    for first_color in range(3):
        for second_color in range(3):
            colored = [None] * len(factors)
            for color, pair in ((first_color, pairing[0]), (second_color, pairing[1])):
                for pos in pair:
                    colored[pos] = color
            expanded[tuple(colored)] += 1
    return [_AssignmentTerm(coeff, tuple(0 for _ in factors), colors) for colors, coeff in sorted(expanded.items())]


def _su3_epsilon_assignments(factors: tuple[FieldFactor, ...]) -> list[_AssignmentTerm]:
    expanded = []
    for colors in _permutations_3():
        sign = _permutation_sign(colors)
        expanded.append(_AssignmentTerm(sp.Integer(sign), tuple(0 for _ in factors), colors))
    return expanded


def _epsilon_coeff(first: int, second: int, third: int) -> sp.Expr:
    if len({first, second, third}) < 3:
        return sp.Integer(0)
    return sp.Integer(_permutation_sign((first, second, third)))


def _sextet_basis() -> tuple[tuple[int, int], ...]:
    return ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))


def _sextet_pair_coeff(first: int, second: int, basis: tuple[int, int]) -> sp.Expr:
    i, j = basis
    if i == j:
        return sp.Integer(1) if first == i and second == j else sp.Integer(0)
    return sp.sqrt(2) / 2 if (first, second) in ((i, j), (j, i)) else sp.Integer(0)


def _sextet_basis_tensor(label: tuple[int, int]) -> dict[tuple[int, int], sp.Expr]:
    return {
        (first, second): _sextet_pair_coeff(first, second, label)
        for first in range(3)
        for second in range(3)
        if _sextet_pair_coeff(first, second, label) != 0
    }


def _su3_sextet_delta_assignments(factors: tuple[FieldFactor, ...]) -> list[_AssignmentTerm]:
    reps = tuple(factor.su3 for factor in factors)
    sextet_pos = next(i for i, rep in enumerate(reps) if rep == SU3Rep(2, 0))
    antisextet_pos = next(i for i, rep in enumerate(reps) if rep == SU3Rep(0, 2))
    terms = []
    for basis in _sextet_basis():
        colors = [None] * len(factors)
        colors[sextet_pos] = basis
        colors[antisextet_pos] = basis
        terms.append(_AssignmentTerm(sp.Integer(1), tuple(0 for _ in factors), tuple(colors)))
    return terms


def _su3_two_triplet_antisextet_assignments(factors: tuple[FieldFactor, ...]) -> list[_AssignmentTerm]:
    reps = tuple(factor.su3 for factor in factors)
    triplet_positions = [i for i, rep in enumerate(reps) if rep == SU3Rep(1, 0)]
    antisextet_pos = next(i for i, rep in enumerate(reps) if rep == SU3Rep(0, 2))
    terms = []
    for basis in _sextet_basis():
        for first in range(3):
            for second in range(3):
                coeff = _sextet_pair_coeff(first, second, basis)
                if coeff == 0:
                    continue
                colors = [None] * len(factors)
                colors[triplet_positions[0]] = first
                colors[triplet_positions[1]] = second
                colors[antisextet_pos] = basis
                terms.append(_AssignmentTerm(coeff, tuple(0 for _ in factors), tuple(colors)))
    return terms


def _su3_two_antitriplet_sextet_assignments(factors: tuple[FieldFactor, ...]) -> list[_AssignmentTerm]:
    reps = tuple(factor.su3 for factor in factors)
    antitriplet_positions = [i for i, rep in enumerate(reps) if rep == SU3Rep(0, 1)]
    sextet_pos = next(i for i, rep in enumerate(reps) if rep == SU3Rep(2, 0))
    terms = []
    for basis in _sextet_basis():
        for first in range(3):
            for second in range(3):
                coeff = _sextet_pair_coeff(first, second, basis)
                if coeff == 0:
                    continue
                colors = [None] * len(factors)
                colors[antitriplet_positions[0]] = first
                colors[antitriplet_positions[1]] = second
                colors[sextet_pos] = basis
                terms.append(_AssignmentTerm(coeff, tuple(0 for _ in factors), tuple(colors)))
    return terms


def _su3_three_sextet_assignments(factors: tuple[FieldFactor, ...]) -> list[_AssignmentTerm]:
    terms = []
    basis = _sextet_basis()
    for first_basis in basis:
        first_tensor = _sextet_basis_tensor(first_basis)
        for second_basis in basis:
            second_tensor = _sextet_basis_tensor(second_basis)
            for third_basis in basis:
                third_tensor = _sextet_basis_tensor(third_basis)
                coeff = sp.Integer(0)
                for (i, j), coeff_a in first_tensor.items():
                    for (k, l), coeff_b in second_tensor.items():
                        for (m, n), coeff_c in third_tensor.items():
                            coeff += (
                                coeff_a
                                * coeff_b
                                * coeff_c
                                * _epsilon_coeff(i, k, m)
                                * _epsilon_coeff(j, l, n)
                            )
                coeff = sp.simplify(coeff)
                if coeff == 0:
                    continue
                terms.append(
                    _AssignmentTerm(
                        coeff,
                        tuple(0 for _ in factors),
                        (first_basis, second_basis, third_basis),
                    )
                )
    return terms



def _delta_tensor_object(left: str, right: str, dimension: int) -> SparseTensorObject:
    return SparseTensorObject.from_entries(
        {(index, index): sp.Integer(1) for index in range(dimension)},
        (left, right),
        (dimension, dimension),
    )


def _rename_tensor_indices(tensor: SparseTensorObject, indices: tuple[str, ...]) -> SparseTensorObject:
    return SparseTensorObject(tensor.entries, indices, tensor.dimensions)


def _delta_pairings(
    reps: tuple[SU3Rep, ...],
    fundamental: SU3Rep,
    antifundamental: SU3Rep,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    fundamentals = [i for i, rep in enumerate(reps) if rep == fundamental]
    antifundamentals = [i for i, rep in enumerate(reps) if rep == antifundamental]
    return [
        ((fundamentals[0], antifundamentals[0]), (fundamentals[1], antifundamentals[1])),
        ((fundamentals[0], antifundamentals[1]), (fundamentals[1], antifundamentals[0])),
    ]


def _is_identical_triplet_antitriplet_quartic(factors: tuple[FieldFactor, ...]) -> bool:
    counts = Counter((factor.field.name, factor.conjugate, factor.su3) for factor in factors)
    if len(counts) != 2 or sorted(counts.values()) != [2, 2]:
        return False
    reps = sorted((key[2] for key in counts), key=lambda rep: (rep.p, rep.q))
    return reps == [SU3Rep(0, 1), SU3Rep(1, 0)]


def _permutations_3() -> list[tuple[int, int, int]]:
    return [(0, 1, 2), (1, 2, 0), (2, 0, 1), (0, 2, 1), (2, 1, 0), (1, 0, 2)]


def _permutation_sign(values: tuple[int, int, int]) -> int:
    inversions = sum(1 for i in range(len(values)) for j in range(i + 1, len(values)) if values[i] > values[j])
    return -1 if inversions % 2 else 1


@dataclass(frozen=True)
class _FactorIndexGroup:
    factor: FieldFactor
    positions: tuple[int, ...]

    @property
    def count(self) -> int:
        return len(self.positions)

def _factor_index_groups(factors: tuple[FieldFactor, ...]) -> list[_FactorIndexGroup]:
    grouped: dict[FieldFactor, list[int]] = defaultdict(list)
    for index, factor in enumerate(factors):
        grouped[factor].append(index)
    return [
        _FactorIndexGroup(factor, tuple(positions))
        for factor, positions in sorted(grouped.items(), key=lambda item: (item[0].field.name, item[0].conjugate))
    ]


def _symmetric_channels(rep: SU2Rep, count: int) -> set[SU2Rep]:
    return set(su2_symmetric_power(rep, count))


def _state_assignment_terms(
    group: _FactorIndexGroup,
    channel: SU2Rep,
    two_m: int,
    length: int,
) -> list[_AssignmentTerm]:
    if group.count == 1:
        if channel != group.factor.su2:
            return []
        weak = [None] * length
        weak[group.positions[0]] = two_m
        return [_AssignmentTerm(sp.Integer(1), tuple(weak), tuple(None for _ in range(length)))]

    if group.count == 2:
        return _two_scalar_state_assignment_terms(group, channel, two_m, length)

    raise NotImplementedError("Expanded SU(2) states currently support up to two identical scalar factors per group")

def _two_scalar_state_assignment_terms(
    group: _FactorIndexGroup,
    channel: SU2Rep,
    two_m: int,
    length: int,
) -> tuple[_AssignmentTerm, ...]:
    return _two_scalar_state_assignment_terms_cached(group, channel, two_m, length)


@lru_cache(maxsize=None)
def _two_scalar_state_assignment_terms_cached(
    group: _FactorIndexGroup,
    channel: SU2Rep,
    two_m: int,
    length: int,
) -> tuple[_AssignmentTerm, ...]:
    rep = group.factor.su2
    allowed = _symmetric_channels(rep, 2)
    if channel not in allowed:
        return ()

    expanded: list[_AssignmentTerm] = []
    pos1, pos2 = group.positions
    pair_weights = _basis_weights(rep, dual=group.factor.conjugate)
    empty_colors = tuple(None for _ in range(length))
    for two_m1 in pair_weights:
        for two_m2 in pair_weights:
            if two_m1 + two_m2 != two_m:
                continue
            coeff = _cg(rep.two_j, two_m1, rep.two_j, two_m2, channel.two_j, two_m)
            if coeff == 0:
                continue
            weak = [None] * length
            weak[pos1] = two_m1
            weak[pos2] = two_m2
            expanded.append(_AssignmentTerm(coeff, tuple(weak), empty_colors))
    return tuple(expanded)

def _rep_generators(rep: SU2Rep, *, dual: bool) -> tuple[sp.Matrix, sp.Matrix, sp.Matrix]:
    j = sp.Rational(rep.two_j, 2)
    weights = _basis_weights(rep, dual=False)
    dim = rep.dim
    j3 = sp.zeros(dim)
    jp = sp.zeros(dim)
    jm = sp.zeros(dim)
    for row, two_m in enumerate(weights):
        m = sp.Rational(two_m, 2)
        j3[row, row] = m
        if row > 0:
            jp[row - 1, row] = sp.sqrt((j - m) * (j + m + 1))
        if row < dim - 1:
            jm[row + 1, row] = sp.sqrt((j + m) * (j - m + 1))
    if not dual:
        return j3, jp, jm
    return -j3.T, -jp.T, -jm.T


def _basis_weights(rep: SU2Rep, *, dual: bool) -> tuple[int, ...]:
    weights = tuple(range(rep.two_j, -rep.two_j - 1, -2))
    if dual:
        return tuple(-weight for weight in weights)
    return weights


def _canonicalize_vector(vector: sp.Matrix | tuple[sp.Expr, ...]) -> tuple[sp.Expr, ...]:
    entries = [_normalize_exact_coeff(entry) for entry in vector]
    pivot = next((entry for entry in entries if entry != 0), None)
    if pivot is None:
        return tuple(sp.Integer(0) for _ in entries)
    return tuple(_normalize_exact_coeff(entry / pivot) for entry in entries)


def _deduplicate_vectors(vectors) -> list[tuple[sp.Expr, ...]]:
    unique: list[tuple[sp.Expr, ...]] = []
    seen: set[tuple[sp.Expr, ...]] = set()
    for vector in vectors:
        if vector in seen:
            continue
        seen.add(vector)
        unique.append(vector)
    return sorted(unique, key=_vector_sort_key)


def _vector_sort_key(vector: tuple[sp.Expr, ...]) -> tuple:
    non_zero = tuple(index for index, value in enumerate(vector) if value != 0)
    return (len(non_zero), non_zero, tuple(str(value) for value in vector))


def _weights(rep: SU2Rep) -> list[int]:
    return list(range(rep.two_j, -rep.two_j - 1, -2))


@lru_cache(maxsize=None)
def _cg(two_j1: int, two_m1: int, two_j2: int, two_m2: int, two_j: int, two_m: int) -> sp.Expr:
    return clebsch_gordan(
        sp.Rational(two_j1, 2),
        sp.Rational(two_j2, 2),
        sp.Rational(two_j, 2),
        sp.Rational(two_m1, 2),
        sp.Rational(two_m2, 2),
        sp.Rational(two_m, 2),
    )


def _component_latex(
    factor: FieldFactor,
    two_m: int,
    color: int | None = None,
    *,
    conjugate_dual_weight: bool = True,
    dual_basis: bool = False,
) -> str:
    # The SU(2) assignment for a conjugated factor lives in the dual basis.
    # The displayed daggered component is the conjugate of the original field
    # component with the opposite T3 weight.
    display_two_m = -two_m if factor.conjugate and conjugate_dual_weight else two_m
    charge = Fraction(display_two_m, 2) + factor.field.hypercharge
    base = field_latex_name(factor.field.name)
    charge_body = _charge_body(charge)
    color_suffix = _color_suffix(factor.su3, color)
    color_superscript = _color_superscript_body(factor.su3, color)
    grouped_base = rf"{{{base}}}"
    if color_superscript is not None:
        component = rf"{grouped_base}^{{{charge_body},{color_superscript}}}"
    else:
        component = rf"{grouped_base}^{{{charge_body}}}{color_suffix}"
    return rf"{{{component}}}^\dagger" if factor.conjugate else component


def _component_phase(factor: FieldFactor, two_m: int, phase_target_fields: tuple[str, ...]) -> sp.Expr:
    """Apply a display-basis phase without changing the invariant construction.

    The target multiplets are inferred from vectorlike fermion pairs at model
    generation time rather than from field names.
    """

    if factor.kind != FieldKind.WEYL_FERMION or factor.conjugate:
        return sp.Integer(1)
    if factor.field.name not in phase_target_fields:
        return sp.Integer(1)
    exponent = (factor.field.su2.two_j + two_m) // 2
    return sp.Integer(-1) if exponent % 2 else sp.Integer(1)


def _component_sort_key(component: tuple[FieldFactor, int, int | str | None]) -> tuple:
    factor, two_m, color = component
    if isinstance(color, tuple):
        color_key = tuple(color)
    else:
        color_key = color
    return (factor.field.name, factor.conjugate, two_m, str(color_key))


def _charge_body(charge: Fraction) -> str:
    if charge == 0:
        return "0"
    if charge == 1:
        return "+"
    if charge == -1:
        return "-"
    if charge > 0 and charge.denominator == 1:
        return "+" * charge.numerator
    if charge < 0 and charge.denominator == 1:
        return "-" * abs(charge.numerator)
    sign = "+" if charge > 0 else ""
    return sign + str(charge)


def _charge_suffix(charge: Fraction) -> str:
    return "^{" + _charge_body(charge) + "}"


def _color_superscript_body(rep: SU3Rep, color: int | None) -> str | None:
    if color is None:
        return None
    labels = ("r", "g", "b")
    if rep == SU3Rep(0, 1):
        return rf"\bar{{{labels[int(color)]}}}"
    if rep == SU3Rep(0, 2):
        i, j = color
        return rf"\overline{{{labels[i] + labels[j]}}}"
    return None


def _color_suffix(rep: SU3Rep, color: int | None) -> str:
    if color is None or rep == SU3Rep(0, 0):
        return ""
    if rep == SU3Rep(1, 1):
        return rf"_{{{color}}}"
    labels = ("r", "g", "b")
    if rep == SU3Rep(1, 0):
        label = labels[int(color)]
        return rf"_{{{label}}}"
    if rep == SU3Rep(0, 1):
        return ""
    if rep in (SU3Rep(2, 0), SU3Rep(0, 2)):
        i, j = color
        label = labels[i] + labels[j]
        if rep == SU3Rep(2, 0):
            return rf"_{{{label}}}"
        return ""
    raise NotImplementedError(f"Component color labels are not implemented for SU(3) rep {rep}")
