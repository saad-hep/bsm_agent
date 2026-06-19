"""Field definitions and conjugated field factors."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
import re

from .groups import SU2Rep, SU3Rep


_FIELD_LATEX_ALIASES: ContextVar[dict[str, str]] = ContextVar("_FIELD_LATEX_ALIASES", default={})


def _normalize_hypercharge(value: Fraction | int | float | str) -> Fraction:
    """Convert common user inputs to the intended exact rational charge."""

    if isinstance(value, Fraction):
        return value.limit_denominator()
    return Fraction(value).limit_denominator()


def field_latex_name(name: str) -> str:
    alias = _FIELD_LATEX_ALIASES.get().get(name)
    if alias:
        return alias
    if name == "q":
        return "q"
    if name == "l":
        return "l"
    match = re.fullmatch(r"([A-Za-z]+)\^C", name)
    if match is not None:
        return rf"{match.group(1)}^{{C}}"
    match = re.fullmatch(r"phi_(\d+)", name)
    if match is not None:
        return rf"\phi_{{{match.group(1)}}}"
    match = re.fullmatch(r"psi_(\d+)", name)
    if match is not None:
        return rf"\psi_{{{match.group(1)}}}"
    match = re.fullmatch(r"Phi(\d+)", name)
    if match is not None:
        return rf"\phi_{{{match.group(1)}}}"
    match = re.fullmatch(r"Psi(\d+)", name)
    if match is not None:
        return rf"\psi_{{{match.group(1)}}}"
    return name


@contextmanager
def field_latex_aliases(aliases: dict[str, str] | None):
    cleaned = {
        str(name).strip(): str(alias).strip()
        for name, alias in (aliases or {}).items()
        if str(name).strip() and str(alias).strip()
    }
    token = _FIELD_LATEX_ALIASES.set(cleaned)
    try:
        yield
    finally:
        _FIELD_LATEX_ALIASES.reset(token)


class FieldKind(str, Enum):
    SCALAR = "scalar"
    WEYL_FERMION = "weyl_fermion"
    GAUGE_BOSON = "gauge_boson"


@dataclass(frozen=True)
class Field:
    name: str
    kind: FieldKind
    su3: SU3Rep
    su2: SU2Rep
    hypercharge: Fraction
    generations: int = 1
    real: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "hypercharge", _normalize_hypercharge(self.hypercharge))

    @classmethod
    def scalar(
        cls,
        name: str,
        *,
        su3: str | int | tuple[int, int] | SU3Rep,
        su2: int,
        hypercharge: Fraction | int | float | str,
        generations: int = 1,
        real: bool = False,
    ) -> "Field":
        return cls(
            name=name,
            kind=FieldKind.SCALAR,
            su3=SU3Rep.parse(su3),
            su2=SU2Rep(su2),
            hypercharge=hypercharge,
            generations=generations,
            real=real,
        )

    @classmethod
    def fermion(
        cls,
        name: str,
        *,
        su3: str | int | tuple[int, int] | SU3Rep,
        su2: int,
        hypercharge: Fraction | int | float | str,
        generations: int = 1,
    ) -> "Field":
        return cls(
            name=name,
            kind=FieldKind.WEYL_FERMION,
            su3=SU3Rep.parse(su3),
            su2=SU2Rep(su2),
            hypercharge=hypercharge,
            generations=generations,
        )

    @property
    def mass_dimension(self) -> Fraction:
        if self.kind == FieldKind.SCALAR:
            return Fraction(1)
        if self.kind == FieldKind.WEYL_FERMION:
            return Fraction(3, 2)
        return Fraction(1)

    def factor(self, conjugate: bool = False) -> "FieldFactor":
        return FieldFactor(self, conjugate and not self.real)


@dataclass(frozen=True, order=True)
class FieldFactor:
    field: Field
    conjugate: bool = False

    @property
    def name(self) -> str:
        return f"{self.field.name}†" if self.conjugate else self.field.name

    @property
    def su3(self) -> SU3Rep:
        return self.field.su3.conjugate if self.conjugate else self.field.su3

    @property
    def su2(self) -> SU2Rep:
        return self.field.su2

    @property
    def hypercharge(self) -> Fraction:
        return -self.field.hypercharge if self.conjugate else self.field.hypercharge

    @property
    def mass_dimension(self) -> Fraction:
        return self.field.mass_dimension

    @property
    def kind(self) -> FieldKind:
        return self.field.kind

    def conjugated(self) -> "FieldFactor":
        return FieldFactor(self.field, not self.conjugate and not self.field.real)

    def latex(self) -> str:
        base = field_latex_name(self.field.name)
        return rf"{base}^\dagger" if self.conjugate else base
