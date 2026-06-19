"""Electroweak symmetry breaking helpers.

This module implements the built-in Standard Model Higgs sector and a generic
symbolic backend for expanding scalar potentials around user-declared VEVs.
It keeps the result symbolic and explicit enough to validate tadpoles, scalar
mass matrices, and tree-level SM relations.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence

import sympy as sp


@dataclass(frozen=True)
class EWSBResult:
    substitutions: dict[str, str]
    scalar_potential: sp.Expr
    tadpoles: dict[str, sp.Expr]
    scalar_masses: dict[str, sp.Expr]
    gauge_boson_masses: dict[str, sp.Expr]
    fermion_masses: dict[str, sp.Expr]
    mixing_relations: dict[str, sp.Expr | str]

    def latex(self) -> str:
        lines = [
            r"\textbf{Field substitutions}",
            *[rf"{lhs} &\to {rhs}" for lhs, rhs in self.substitutions.items()],
            r"\textbf{Scalar potential}",
            rf"V &= {sp.latex(self.scalar_potential)}",
            r"\textbf{Tadpoles}",
            *[rf"{name} &: {sp.latex(value)}" for name, value in self.tadpoles.items()],
            r"\textbf{Scalar masses}",
            *[rf"{name} &: {sp.latex(value)}" for name, value in self.scalar_masses.items()],
            r"\textbf{Gauge boson masses}",
            *[rf"{name} &: {sp.latex(value)}" for name, value in self.gauge_boson_masses.items()],
            r"\textbf{Fermion masses}",
            *[rf"{name} &: {sp.latex(value)}" for name, value in self.fermion_masses.items()],
            r"\textbf{Mixing relations}",
            *[rf"{name} &: {sp.latex(value) if not isinstance(value, str) else value}" for name, value in self.mixing_relations.items()],
        ]
        return r" \\" + "\n".join(lines)


@dataclass(frozen=True)
class VEVExpansion:
    """A scalar field shift used by the generic EWSB backend.

    A real scalar convention is

        S -> v + s.

    A complex neutral scalar convention is

        Phi  -> (v + h + i a)/sqrt(2),
        Phi* -> (v + h - i a)/sqrt(2).

    Use :meth:`real_scalar` and :meth:`complex_scalar` for the standard
    conventions.  The lower-level constructor also supports custom
    normalizations for non-canonical input variables.
    """

    field: sp.Symbol
    vev: sp.Symbol | sp.Integer
    real_part: sp.Symbol
    conjugate: sp.Symbol | None = None
    imag_part: sp.Symbol | None = None
    normalization: sp.Expr = sp.Integer(1)

    @classmethod
    def real_scalar(cls, field: sp.Symbol, vev: sp.Symbol | sp.Integer, fluctuation: sp.Symbol) -> "VEVExpansion":
        return cls(field=field, vev=vev, real_part=fluctuation)

    @classmethod
    def complex_scalar(
        cls,
        field: sp.Symbol,
        conjugate: sp.Symbol,
        vev: sp.Symbol | sp.Integer,
        cp_even: sp.Symbol,
        cp_odd: sp.Symbol,
    ) -> "VEVExpansion":
        return cls(
            field=field,
            conjugate=conjugate,
            vev=vev,
            real_part=cp_even,
            imag_part=cp_odd,
            normalization=sp.sqrt(2) ** -1,
        )

    @property
    def fluctuations(self) -> tuple[sp.Symbol, ...]:
        if self.imag_part is None:
            return (self.real_part,)
        return (self.real_part, self.imag_part)

    def substitutions(self) -> dict[sp.Symbol, sp.Expr]:
        real_shift = self.normalization * (self.vev + self.real_part)
        if self.imag_part is None:
            substitutions = {self.field: real_shift}
            if self.conjugate is not None:
                substitutions[self.conjugate] = real_shift
            return substitutions

        if self.conjugate is None:
            raise ValueError("Complex VEV expansions require a conjugate symbol")
        return {
            self.field: self.normalization * (self.vev + self.real_part + sp.I * self.imag_part),
            self.conjugate: self.normalization * (self.vev + self.real_part - sp.I * self.imag_part),
        }


@dataclass(frozen=True)
class GenericEWSBResult:
    substitutions: dict[sp.Symbol, sp.Expr]
    shifted_potential: sp.Expr
    vacuum_potential: sp.Expr
    fields: tuple[sp.Symbol, ...]
    tadpoles: dict[sp.Symbol, sp.Expr]
    mass_matrix: sp.Matrix
    tadpole_solution: dict[sp.Symbol, sp.Expr]
    solved_vacuum_potential: sp.Expr
    solved_mass_matrix: sp.Matrix

    def latex(self) -> str:
        matrix_latex = sp.latex(self.solved_mass_matrix if self.tadpole_solution else self.mass_matrix)
        lines = [
            r"\textbf{Field substitutions}",
            *[rf"{sp.latex(lhs)} &\to {sp.latex(rhs)}" for lhs, rhs in self.substitutions.items()],
            r"\textbf{Shifted scalar potential}",
            rf"V &= {sp.latex(self.shifted_potential)}",
            r"\textbf{Tadpoles}",
            *[rf"\frac{{\partial V}}{{\partial {sp.latex(field)}}}\bigg|_0 &= {sp.latex(value)}" for field, value in self.tadpoles.items()],
            r"\textbf{Scalar mass matrix}",
            rf"M^2_{{{','.join(sp.latex(field) for field in self.fields)}}} &= {matrix_latex}",
        ]
        if self.tadpole_solution:
            lines.extend(
                [
                    r"\textbf{Tadpole solution}",
                    *[rf"{sp.latex(lhs)} &= {sp.latex(rhs)}" for lhs, rhs in self.tadpole_solution.items()],
                ]
            )
        return r" \\" + "\n".join(lines)


def expand_around_vevs(
    potential: sp.Expr,
    expansions: Sequence[VEVExpansion],
    *,
    zero_substitutions: Mapping[sp.Symbol, sp.Expr] | None = None,
    solve_for: Sequence[sp.Symbol] = (),
) -> GenericEWSBResult:
    """Expand a scalar potential around declared VEVs.

    Parameters
    ----------
    potential:
        Symbolic expression for the scalar potential before shifting fields.
    expansions:
        VEV declarations.  Each declaration supplies the field substitution and
        the real fluctuation fields used for tadpoles and mass matrices.
    zero_substitutions:
        Optional additional substitutions applied at the vacuum after setting
        fluctuation fields to zero.
    solve_for:
        Optional symbols to eliminate with the tadpole equations.  The first
        symbolic solution returned by the solver is used.
    """

    substitutions: dict[sp.Symbol, sp.Expr] = {}
    fields: list[sp.Symbol] = []
    for expansion in expansions:
        overlap = set(substitutions).intersection(expansion.substitutions())
        if overlap:
            names = ", ".join(str(symbol) for symbol in sorted(overlap, key=str))
            raise ValueError(f"Duplicate VEV substitution for: {names}")
        substitutions.update(expansion.substitutions())
        for field in expansion.fluctuations:
            if field not in fields:
                fields.append(field)

    shifted = sp.expand(sp.sympify(potential).subs(substitutions))
    vacuum_subs: dict[sp.Symbol, sp.Expr] = {field: sp.Integer(0) for field in fields}
    if zero_substitutions:
        vacuum_subs.update(dict(zero_substitutions))

    vacuum_potential = sp.simplify(shifted.subs(vacuum_subs))
    tadpoles = {field: sp.simplify(sp.diff(shifted, field).subs(vacuum_subs)) for field in fields}
    mass_matrix = sp.Matrix(
        [[sp.simplify(sp.diff(shifted, left, right).subs(vacuum_subs)) for right in fields] for left in fields]
    )

    tadpole_solution: dict[sp.Symbol, sp.Expr] = {}
    if solve_for:
        solutions = sp.solve(list(tadpoles.values()), list(solve_for), dict=True, simplify=True)
        if solutions:
            tadpole_solution = {symbol: sp.simplify(value) for symbol, value in solutions[0].items()}

    solved_vacuum = sp.simplify(vacuum_potential.subs(tadpole_solution))
    solved_mass_matrix = sp.simplify(mass_matrix.subs(tadpole_solution))

    return GenericEWSBResult(
        substitutions=substitutions,
        shifted_potential=shifted,
        vacuum_potential=vacuum_potential,
        fields=tuple(fields),
        tadpoles=tadpoles,
        mass_matrix=mass_matrix,
        tadpole_solution=tadpole_solution,
        solved_vacuum_potential=solved_vacuum,
        solved_mass_matrix=solved_mass_matrix,
    )


def sm_ewsb() -> EWSBResult:
    r"""Return tree-level SM EWSB formulas.

    The Higgs convention is

    H = (G^+, (v + h + i G^0)/sqrt(2)).

    The potential convention is

    V = -mu2 H^\dagger H + (lambda/2) (H^\dagger H)^2.
    """

    v, h, g0, gp, gm = sp.symbols("v h G0 Gp Gm", real=True)
    mu2, lam = sp.symbols("mu2 lambda", real=True)
    g1, g2 = sp.symbols("g1 g2", real=True)
    yu, yd, ye = sp.symbols("Y_u Y_d Y_e")

    hdagh = gm * gp + ((v + h) ** 2 + g0**2) / 2
    potential = sp.expand(-mu2 * hdagh + sp.Rational(1, 2) * lam * hdagh**2)
    vacuum_potential = sp.expand(potential.subs({h: 0, g0: 0, gp: 0, gm: 0}))
    tadpole_h = sp.diff(potential, h).subs({h: 0, g0: 0, gp: 0, gm: 0})

    minimum_rule = {mu2: lam * v**2 / 2}
    scalar_masses = {
        "m_h^2": sp.simplify(sp.diff(potential, h, h).subs({h: 0, g0: 0, gp: 0, gm: 0}).subs(minimum_rule)),
        "m_G0^2": sp.simplify(sp.diff(potential, g0, g0).subs({h: 0, g0: 0, gp: 0, gm: 0}).subs(minimum_rule)),
        "m_G+G-^2": sp.simplify(sp.diff(sp.diff(potential, gp), gm).subs({h: 0, g0: 0, gp: 0, gm: 0}).subs(minimum_rule)),
    }

    gauge_boson_masses = {
        "m_W^2": sp.simplify(g2**2 * v**2 / 4),
        "m_Z^2": sp.simplify((g1**2 + g2**2) * v**2 / 4),
        "m_A^2": sp.Integer(0),
    }

    fermion_masses = {
        "m_u": yu * v / sp.sqrt(2),
        "m_d": yd * v / sp.sqrt(2),
        "m_e": ye * v / sp.sqrt(2),
    }

    mixing_relations = {
        "tan(theta_W)": g1 / g2,
        "A_mu": r"\cos\theta_W B_\mu + \sin\theta_W W^3_\mu",
        "Z_mu": r"-\sin\theta_W B_\mu + \cos\theta_W W^3_\mu",
        "W^pm_mu": r"(W^1_\mu \mp i W^2_\mu)/\sqrt{2}",
    }

    return EWSBResult(
        substitutions={
            "H^+": "G^+",
            "H^0": r"(v + h + i G^0)/\sqrt{2}",
            "{H^0}^dagger": r"(v + h - i G^0)/\sqrt{2}",
        },
        scalar_potential=potential,
        tadpoles={"dV/dh|0": sp.simplify(tadpole_h), "minimum": sp.Eq(mu2, lam * v**2 / 2)},
        scalar_masses=scalar_masses,
        gauge_boson_masses=gauge_boson_masses,
        fermion_masses=fermion_masses,
        mixing_relations=mixing_relations,
    )
