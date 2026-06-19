"""Small SM-gauge group theory backend.

The backend deliberately focuses on the Standard Model gauge group.  It supports
the low-dimensional representations that cover most first-pass BSM extensions.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
import re
from typing import Iterable

@dataclass(frozen=True, order=True)
class SU2Rep:
    """SU(2) irrep represented by its dimension."""

    dim: int

    def __post_init__(self) -> None:
        if self.dim < 1:
            raise ValueError("SU(2) representation dimension must be positive")

    @property
    def two_j(self) -> int:
        return self.dim - 1

    @property
    def conjugate(self) -> "SU2Rep":
        return self

    def __str__(self) -> str:
        return str(self.dim)


@dataclass(frozen=True, order=True)
class SU3Rep:
    """SU(3) irrep represented by Dynkin labels (p, q)."""

    p: int
    q: int

    def __post_init__(self) -> None:
        if self.p < 0 or self.q < 0:
            raise ValueError("SU(3) Dynkin labels must be non-negative")

    @property
    def conjugate(self) -> "SU3Rep":
        return SU3Rep(self.q, self.p)

    @property
    def dimension(self) -> int:
        return (self.p + 1) * (self.q + 1) * (self.p + self.q + 2) // 2

    @classmethod
    def parse(cls, value: str | int | tuple[int, int] | "SU3Rep") -> "SU3Rep":
        if isinstance(value, SU3Rep):
            return value
        if isinstance(value, tuple):
            return cls(*value)

        text = str(value).strip().lower()
        if not text:
            raise ValueError(f"Unsupported SU(3) representation: {value!r}")

        normalized = re.sub(r"[\s_]+", "", text).replace("−", "-")
        if normalized in {"-3", "-6", "-10"}:
            normalized = {
                "-3": "bar3",
                "-6": "bar6",
                "-10": "bar10",
            }[normalized]
        else:
            normalized = normalized.replace("-", "")

        table = {
            "1": cls(0, 0),
            "singlet": cls(0, 0),
            "3": cls(1, 0),
            "triplet": cls(1, 0),
            "fundamental": cls(1, 0),
            "bar3": cls(0, 1),
            "3bar": cls(0, 1),
            "3*": cls(0, 1),
            "3star": cls(0, 1),
            "anti3": cls(0, 1),
            "antitriplet": cls(0, 1),
            "antifundamental": cls(0, 1),
            "6": cls(2, 0),
            "sextet": cls(2, 0),
            "bar6": cls(0, 2),
            "6bar": cls(0, 2),
            "6*": cls(0, 2),
            "6star": cls(0, 2),
            "anti6": cls(0, 2),
            "antisextet": cls(0, 2),
            "8": cls(1, 1),
            "octet": cls(1, 1),
            "10": cls(3, 0),
            "decuplet": cls(3, 0),
            "bar10": cls(0, 3),
            "10bar": cls(0, 3),
            "10*": cls(0, 3),
            "10star": cls(0, 3),
            "anti10": cls(0, 3),
            "antidecuplet": cls(0, 3),
        }
        try:
            return table[normalized]
        except KeyError as exc:
            raise ValueError(f"Unsupported SU(3) representation: {value!r}") from exc

    def label(self) -> str:
        table = {
            (0, 0): "1",
            (1, 0): "3",
            (0, 1): "bar3",
            (2, 0): "6",
            (0, 2): "bar6",
            (1, 1): "8",
            (3, 0): "10",
            (0, 3): "bar10",
        }
        return table.get((self.p, self.q), f"({self.p},{self.q})")

    def latex(self) -> str:
        table = {
            (0, 0): "1",
            (1, 0): "3",
            (0, 1): r"\overline{3}",
            (2, 0): "6",
            (0, 2): r"\overline{6}",
            (1, 1): "8",
            (3, 0): "10",
            (0, 3): r"\overline{10}",
        }
        return table.get((self.p, self.q), rf"({self.p},{self.q})")

    def __str__(self) -> str:
        return self.label()


def parse_hypercharge(value: int | Fraction) -> Fraction:
    return value if isinstance(value, Fraction) else Fraction(value)


def su2_product_pair(a: SU2Rep, b: SU2Rep) -> set[SU2Rep]:
    """Decompose a pair of SU(2) irreps.

    Internally this uses twice-isospin integers, so the standard angular
    momentum product is exact.
    """

    lo = abs(a.two_j - b.two_j)
    hi = a.two_j + b.two_j
    return {SU2Rep(two_j + 1) for two_j in range(lo, hi + 1, 2)}


def su2_product_pair_counter(a: SU2Rep, b: SU2Rep) -> Counter[SU2Rep]:
    return Counter(su2_product_pair(a, b))


def su2_product_counter(reps: Iterable[SU2Rep]) -> Counter[SU2Rep]:
    reps = list(reps)
    if not reps:
        return Counter({SU2Rep(1): 1})
    result = Counter({reps[0]: 1})
    for rep in reps[1:]:
        next_result: Counter[SU2Rep] = Counter()
        for current, current_mult in result.items():
            for out, out_mult in su2_product_pair_counter(current, rep).items():
                next_result[out] += current_mult * out_mult
        result = next_result
    return result


def su2_product(reps: Iterable[SU2Rep]) -> set[SU2Rep]:
    return set(su2_product_counter(reps))


def su2_symmetric_power(rep: SU2Rep, power: int) -> Counter[SU2Rep]:
    """Decompose Sym^power(rep) for SU(2).

    This uses the character's weight multiplicities.  The implementation is
    compact and exact for the small powers needed by renormalizable scalar
    operators.
    """

    if power < 0:
        raise ValueError("Symmetric power must be non-negative")
    if power == 0:
        return Counter({SU2Rep(1): 1})
    weights = list(range(-rep.two_j, rep.two_j + 1, 2))
    totals: Counter[int] = Counter({0: 1})
    for weight in weights:
        updated: Counter[int] = Counter()
        for used in range(power + 1):
            for total, multiplicity in totals.items():
                updated[total + used * weight] += multiplicity
        totals = updated
    # Keep only monomials of total degree exactly `power`.
    degree_totals: Counter[tuple[int, int]] = Counter({(0, 0): 1})
    for weight in weights:
        updated = Counter()
        for (degree, total), multiplicity in degree_totals.items():
            for used in range(power - degree + 1):
                updated[(degree + used, total + used * weight)] += multiplicity
        degree_totals = updated
    remaining = Counter({total: mult for (degree, total), mult in degree_totals.items() if degree == power})
    irreps: Counter[SU2Rep] = Counter()
    while remaining:
        highest = max(weight for weight, mult in remaining.items() if mult > 0)
        multiplicity = remaining[highest]
        irrep = SU2Rep(highest + 1)
        irreps[irrep] += multiplicity
        for weight in range(-highest, highest + 1, 2):
            remaining[weight] -= multiplicity
            if remaining[weight] == 0:
                del remaining[weight]
            elif remaining[weight] < 0:
                raise RuntimeError("Invalid SU(2) symmetric-power decomposition")
    return irreps


def su2_singlet_multiplicity(reps: Iterable[SU2Rep]) -> int:
    return su2_product_counter(reps)[SU2Rep(1)]


def su3_product_pair_counter(a: SU3Rep, b: SU3Rep) -> Counter[SU3Rep]:
    """Decompose common low-dimensional SU(3) products.

    The table is intentionally compact. Unsupported products fail loudly, which
    keeps generated physics from silently exceeding the backend's validation.
    """

    table: dict[tuple[tuple[int, int], tuple[int, int]], Counter[tuple[int, int]]] = {
        ((0, 0), (0, 0)): Counter({(0, 0): 1}),
        ((0, 0), (1, 0)): Counter({(1, 0): 1}),
        ((0, 0), (0, 1)): Counter({(0, 1): 1}),
        ((0, 0), (2, 0)): Counter({(2, 0): 1}),
        ((0, 0), (0, 2)): Counter({(0, 2): 1}),
        ((0, 0), (1, 1)): Counter({(1, 1): 1}),
        ((1, 0), (0, 1)): Counter({(0, 0): 1, (1, 1): 1}),
        ((1, 0), (1, 0)): Counter({(2, 0): 1, (0, 1): 1}),
        ((0, 1), (0, 1)): Counter({(0, 2): 1, (1, 0): 1}),
        ((1, 1), (1, 1)): Counter({(0, 0): 1, (1, 1): 2, (3, 0): 1, (0, 3): 1, (2, 2): 1}),
        ((1, 1), (1, 0)): Counter({(1, 0): 1, (0, 2): 1, (2, 1): 1}),
        ((1, 1), (0, 1)): Counter({(0, 1): 1, (2, 0): 1, (1, 2): 1}),
        ((2, 0), (0, 2)): Counter({(0, 0): 1, (1, 1): 1, (2, 2): 1}),
        ((2, 0), (2, 0)): Counter({(0, 2): 1, (2, 1): 1, (4, 0): 1}),
        ((0, 2), (0, 2)): Counter({(2, 0): 1, (1, 2): 1, (0, 4): 1}),
        ((2, 0), (1, 0)): Counter({(3, 0): 1, (1, 1): 1}),
        ((0, 2), (0, 1)): Counter({(0, 3): 1, (1, 1): 1}),
        ((2, 0), (0, 1)): Counter({(1, 0): 1, (2, 1): 1}),
        ((0, 2), (1, 0)): Counter({(0, 1): 1, (1, 2): 1}),
        ((2, 1), (0, 1)): Counter({(2, 2): 1, (3, 0): 1, (1, 1): 1}),
        ((1, 2), (1, 0)): Counter({(2, 2): 1, (0, 3): 1, (1, 1): 1}),
    }
    left = (a.p, a.q)
    right = (b.p, b.q)
    try:
        labels = table[(left, right)] if (left, right) in table else table[(right, left)]
        return Counter({SU3Rep(*rep): mult for rep, mult in labels.items()})
    except KeyError as exc:
        raise NotImplementedError(
            f"SU(3) product {SU3Rep(*left)} x {SU3Rep(*right)} is not implemented"
        ) from exc


def su3_product_pair(a: SU3Rep, b: SU3Rep) -> set[SU3Rep]:
    return set(su3_product_pair_counter(a, b))


def su3_product_counter(reps: Iterable[SU3Rep]) -> Counter[SU3Rep]:
    reps = list(reps)
    if not reps:
        return Counter({SU3Rep(0, 0): 1})
    result = Counter({reps[0]: 1})
    for rep in reps[1:]:
        next_result: Counter[SU3Rep] = Counter()
        for current, current_mult in result.items():
            for out, out_mult in su3_product_pair_counter(current, rep).items():
                next_result[out] += current_mult * out_mult
        result = next_result
    return result


def su3_product(reps: Iterable[SU3Rep]) -> set[SU3Rep]:
    return set(su3_product_counter(reps))


def su3_singlet_multiplicity(reps: Iterable[SU3Rep]) -> int:
    from .su3_invariants import singlet_multiplicity_for_reps as su3_general_singlet_multiplicity

    return su3_general_singlet_multiplicity(tuple(reps))


@dataclass(frozen=True)
class SMGaugeBackend:
    """Gauge-invariance checks for SU(3)c x SU(2)L x U(1)Y."""

    def is_singlet(
        self,
        su3_reps: Iterable[SU3Rep],
        su2_reps: Iterable[SU2Rep],
        hypercharges: Iterable[Fraction],
    ) -> bool:
        return self.singlet_multiplicity(su3_reps, su2_reps, hypercharges) > 0

    def singlet_multiplicity(
        self,
        su3_reps: Iterable[SU3Rep],
        su2_reps: Iterable[SU2Rep],
        hypercharges: Iterable[Fraction],
    ) -> int:
        return (
            su2_singlet_multiplicity(su2_reps) * su3_singlet_multiplicity(su3_reps)
            if sum(hypercharges, Fraction(0)) == 0
            else 0
        )
