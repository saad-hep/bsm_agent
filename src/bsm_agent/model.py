"""Model container and renormalizable operator generation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations_with_replacement

from .fields import Field, FieldFactor, FieldKind
from .groups import SMGaugeBackend, SU2Rep, SU3Rep
from .operators import Lagrangian, Operator
from .scalar_invariants import singlet_multiplicity as scalar_singlet_multiplicity

_SM_FERMION_NAMES = frozenset({"q", "l", "d^C", "u^C", "e^C"})
_SCALAR_POTENTIAL_PREFIXES = {
    1: "xi3",
    2: "m2",
    3: "mu",
    4: "lambda",
}


def _canonical_factors(factors: tuple[FieldFactor, ...]) -> tuple[FieldFactor, ...]:
    return tuple(sorted(factors, key=lambda f: (f.field.kind.value, f.field.name, f.conjugate)))


def _fermion_order_key(factor: FieldFactor) -> tuple:
    return (
        0 if factor.su2 == SU2Rep(1) else 1,
        factor.su2.dim,
        factor.su3.dimension,
        factor.field.name,
        factor.conjugate,
    )


def _ordered_operator_factors(category: str, factors: tuple[FieldFactor, ...], *, preserve_scalar_order: bool = False) -> tuple[FieldFactor, ...]:
    if category == "scalar_potential":
        if preserve_scalar_order:
            return factors
        return _canonical_factors(factors)

    if category == "fermion_mass":
        fermions = tuple(factor for factor in factors if factor.kind == FieldKind.WEYL_FERMION)
        return tuple(sorted(fermions, key=_fermion_order_key))

    if category == "yukawa":
        fermions = tuple(factor for factor in factors if factor.kind == FieldKind.WEYL_FERMION)
        scalars = tuple(factor for factor in factors if factor.kind == FieldKind.SCALAR)
        if len(fermions) == 2 and len(scalars) == 1:
            ordered_fermions = tuple(sorted(fermions, key=_fermion_order_key))
            scalar = scalars[0]
            return (scalar, *ordered_fermions) if scalar.conjugate else (*ordered_fermions, scalar)

    return factors


def _phase_target_fields_for_fermion_pairs(fields: tuple[Field, ...]) -> tuple[str, ...]:
    targets: list[str] = []
    seen_pairs: set[frozenset[str]] = set()
    for left_index, left in enumerate(fields):
        if left.kind != FieldKind.WEYL_FERMION or left.su2.dim <= 1:
            continue
        if left.name in _SM_FERMION_NAMES:
            continue
        for right_index in range(left_index + 1, len(fields)):
            right = fields[right_index]
            if right.kind != FieldKind.WEYL_FERMION:
                continue
            if right.name in _SM_FERMION_NAMES:
                continue
            if left.su2 != right.su2:
                continue
            if left.su3.conjugate != right.su3:
                continue
            if left.hypercharge != -right.hypercharge:
                continue
            pair_key = frozenset((left.name, right.name))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            targets.append(right.name)
    return tuple(targets)


def _dual_basis_slots_for_operator(
    category: str,
    ordered: tuple[FieldFactor, ...],
) -> tuple[int, ...]:
    if category not in {"fermion_mass", "yukawa"}:
        return ()
    fermion_positions = [index for index, factor in enumerate(ordered) if factor.kind == FieldKind.WEYL_FERMION]
    if len(fermion_positions) != 2:
        return ()
    first = ordered[fermion_positions[0]]
    second = ordered[fermion_positions[1]]
    if first.field != second.field:
        return ()
    if first.conjugate or second.conjugate:
        return ()
    if first.field.hypercharge != 0:
        return ()
    return (fermion_positions[1],)


def _operator_dedupe_key(
    category: str,
    factors: tuple[FieldFactor, ...],
) -> tuple:
    if category == "scalar_potential":
        return tuple((f.field.name, f.conjugate) for f in _canonical_factors(factors))
    return tuple((f.field.name, f.conjugate) for f in factors)


@dataclass(frozen=True)
class AnomalyReport:
    su3_cubic: Fraction
    su2_su2_u1: Fraction
    su3_su3_u1: Fraction
    u1_gravity: Fraction
    u1_cubic: Fraction

    @property
    def is_anomaly_free(self) -> bool:
        return all(value == 0 for value in self.__dict__.values())

    def summary(self) -> str:
        status = "anomaly-free" if self.is_anomaly_free else "anomalous"
        values = ", ".join(f"{key}={value}" for key, value in self.__dict__.items())
        return f"{status}: {values}"


@dataclass(frozen=True)
class Model:
    name: str
    fields: tuple[Field, ...]
    backend: SMGaugeBackend = SMGaugeBackend()
    gauge_kinetic_terms: tuple[str, ...] = (
        "-1/4 B_mn B^mn",
        "-1/4 W_mn^a W^{a,mn}",
        "-1/4 G_mn^A G^{A,mn}",
    )

    def extend(self, fields: list[Field] | tuple[Field, ...], name: str | None = None) -> "Model":
        existing = {field.name for field in self.fields}
        overlap = existing.intersection(field.name for field in fields)
        if overlap:
            raise ValueError(f"Duplicate field names: {sorted(overlap)}")
        return Model(
            name or f"{self.name}+BSM",
            self.fields + tuple(fields),
            self.backend,
            self.gauge_kinetic_terms,
        )

    @property
    def scalars(self) -> tuple[Field, ...]:
        return tuple(field for field in self.fields if field.kind == FieldKind.SCALAR)

    @property
    def fermions(self) -> tuple[Field, ...]:
        return tuple(field for field in self.fields if field.kind == FieldKind.WEYL_FERMION)

    def field_factors(self, kind: FieldKind | None = None) -> tuple[FieldFactor, ...]:
        selected = self.fields if kind is None else tuple(f for f in self.fields if f.kind == kind)
        factors: list[FieldFactor] = []
        for field in selected:
            factors.append(field.factor(False))
            if not field.real and field.kind == FieldKind.SCALAR:
                factors.append(field.factor(True))
        return tuple(factors)

    def is_gauge_singlet(self, factors: tuple[FieldFactor, ...]) -> bool:
        return self.invariant_multiplicity(factors) > 0

    def invariant_multiplicity(self, factors: tuple[FieldFactor, ...]) -> int:
        if all(factor.kind == FieldKind.SCALAR for factor in factors):
            return _scalar_invariant_multiplicity(factors)
        return self.backend.singlet_multiplicity(
            [factor.su3 for factor in factors],
            [factor.su2 for factor in factors],
            [factor.hypercharge for factor in factors],
        )

    def generate_lagrangian(self, max_dimension: int = 4) -> Lagrangian:
        if max_dimension != 4:
            raise NotImplementedError("Only renormalizable dimension <= 4 generation is implemented")
        operators: list[Operator] = []
        seen: set[tuple] = set()
        phase_target_fields = frozenset(_phase_target_fields_for_fermion_pairs(self.fermions))

        def add_operator(
            factors: tuple[FieldFactor, ...],
            category: str,
            prefix: str,
            add_hc: bool,
            multiplicity: int,
        ) -> None:
            ordered = _ordered_operator_factors(category, factors)
            key = (
                category,
                _operator_dedupe_key(category, ordered),
            )
            hc_key = (
                category,
                _operator_dedupe_key(category, tuple(f.conjugated() for f in ordered)),
            )
            if key in seen or hc_key in seen:
                return
            seen.add(key)
            base_index = len([op for op in operators if op.category == category]) + 1
            for contraction_index in range(1, multiplicity + 1):
                coefficient = (
                    f"{prefix}_{base_index}"
                    if multiplicity == 1
                    else f"{prefix}_{base_index}_c{contraction_index}"
                )
                operators.append(
                    Operator(
                        ordered,
                        category,
                        coefficient,
                        add_hc=add_hc,
                        contraction_index=contraction_index,
                        contraction_count=multiplicity,
                        phase_target_fields=tuple(
                            factor.field.name
                            for factor in ordered
                            if factor.field.name in phase_target_fields
                        ),
                        dual_basis_slots=_dual_basis_slots_for_operator(category, ordered),
                    )
                )

        scalar_factors = self.field_factors(FieldKind.SCALAR)
        fermion_factors = tuple(field.factor(False) for field in self.fermions)

        for degree in range(1, 5):
            for combo in combinations_with_replacement(scalar_factors, degree):
                multiplicity = self.invariant_multiplicity(combo)
                if sum((f.mass_dimension for f in combo), Fraction(0)) <= max_dimension and multiplicity > 0:
                    add_operator(
                        combo,
                        "scalar_potential",
                        _SCALAR_POTENTIAL_PREFIXES[degree],
                        add_hc=not _is_self_conjugate(combo),
                        multiplicity=multiplicity,
                    )

        for combo in combinations_with_replacement(fermion_factors, 2):
            multiplicity = self.invariant_multiplicity(combo)
            if multiplicity > 0:
                add_operator(combo, "fermion_mass", "M", add_hc=not _is_self_conjugate(combo), multiplicity=multiplicity)

        for fermions in combinations_with_replacement(fermion_factors, 2):
            for scalar in scalar_factors:
                combo = (*fermions, scalar)
                multiplicity = self.invariant_multiplicity(combo)
                if multiplicity > 0:
                    add_operator(combo, "yukawa", "Y", add_hc=True, multiplicity=multiplicity)

        kinetic_terms = self.gauge_kinetic_terms + tuple(f"D[{field.name}]† D[{field.name}]" for field in self.fields)
        return Lagrangian(kinetic_terms=kinetic_terms, operators=tuple(operators))

    def gauge_interactions_latex(self) -> str:
        from .interactions import gauge_interactions_latex

        return " \\\\\n".join(gauge_interactions_latex(field) for field in self.fields)

    def scalar_seagulls_latex(self) -> str:
        from .interactions import scalar_seagull_latex

        return " \\\\\n".join(scalar_seagull_latex(field) for field in self.scalars)

    def ewsb(self):
        if self.name != "SM":
            raise NotImplementedError("Automatic EWSB is currently implemented for the built-in SM only")
        from .ewsb import sm_ewsb

        return sm_ewsb()

    def expand_around_vevs(self, potential, expansions, *, zero_substitutions=None, solve_for=()):
        """Expand a symbolic scalar potential around user-declared VEVs."""

        from .ewsb import expand_around_vevs

        return expand_around_vevs(
            potential,
            expansions,
            zero_substitutions=zero_substitutions,
            solve_for=solve_for,
        )

    def anomaly_report(self) -> AnomalyReport:
        """Return simple anomaly diagnostics for chiral Weyl fermions.

        The Dynkin normalizations are conventional up to a common factor; zero
        is the meaningful result here.
        """

        su3_cubic = Fraction(0)
        su2_su2_u1 = Fraction(0)
        su3_su3_u1 = Fraction(0)
        u1_gravity = Fraction(0)
        u1_cubic = Fraction(0)
        for field in self.fermions:
            mult = field.generations
            su2_dim = field.su2.dim
            su3_dim = field.su3.dimension
            y = field.hypercharge
            su3_cubic += mult * su2_dim * _su3_cubic_index(field.su3.p, field.su3.q)
            su2_su2_u1 += mult * su3_dim * _su2_dynkin(field.su2.dim) * y
            su3_su3_u1 += mult * su2_dim * _su3_dynkin(field.su3.p, field.su3.q) * y
            u1_gravity += mult * su2_dim * su3_dim * y
            u1_cubic += mult * su2_dim * su3_dim * y**3
        return AnomalyReport(su3_cubic, su2_su2_u1, su3_su3_u1, u1_gravity, u1_cubic)


def _is_self_conjugate(factors: tuple[FieldFactor, ...]) -> bool:
    lhs = sorted((f.field.name, f.conjugate) for f in factors)
    rhs = sorted((f.field.name, f.conjugated().conjugate) for f in factors)
    return lhs == rhs


def _scalar_invariant_multiplicity(factors: tuple[FieldFactor, ...]) -> int:
    return scalar_singlet_multiplicity(factors)


def _su2_dynkin(dim: int) -> Fraction:
    # T(j) = (1/3) j(j+1)(2j+1), with 2j = dim - 1.
    two_j = dim - 1
    return Fraction(two_j * (two_j + 2) * dim, 12)


def _su3_dynkin(p: int, q: int) -> Fraction:
    known = {
        (0, 0): Fraction(0),
        (1, 0): Fraction(1, 2),
        (0, 1): Fraction(1, 2),
        (2, 0): Fraction(5, 2),
        (0, 2): Fraction(5, 2),
        (1, 1): Fraction(3),
    }
    return known.get((p, q), Fraction(0))


def _su3_cubic_index(p: int, q: int) -> Fraction:
    known = {
        (0, 0): Fraction(0),
        (1, 0): Fraction(1),
        (0, 1): Fraction(-1),
        (2, 0): Fraction(7),
        (0, 2): Fraction(-7),
        (1, 1): Fraction(0),
    }
    return known.get((p, q), Fraction(0))
