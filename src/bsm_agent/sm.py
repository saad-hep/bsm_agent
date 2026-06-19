"""Built-in Standard Model in left-handed Weyl notation."""

from __future__ import annotations

from fractions import Fraction

from .fields import Field
from .model import Model


def StandardModel() -> Model:
    """Return the SM field content before EWSB.

    Fermions use left-handed notation:

    q = (u_L, d_L), l = (nu_L, e_L), dC = d_R^c, uC = u_R^c, eC = e_R^c.
    """

    fields = (
        Field.fermion("q", generations=3, su3="3", su2=2, hypercharge=Fraction(1, 6)),
        Field.fermion("l", generations=3, su3="1", su2=2, hypercharge=Fraction(-1, 2)),
        Field.fermion("d^C", generations=3, su3="bar3", su2=1, hypercharge=Fraction(1, 3)),
        Field.fermion("u^C", generations=3, su3="bar3", su2=1, hypercharge=Fraction(-2, 3)),
        Field.fermion("e^C", generations=3, su3="1", su2=1, hypercharge=Fraction(1)),
        Field.scalar("H", generations=1, su3="1", su2=2, hypercharge=Fraction(1, 2)),
    )
    return Model("SM", fields)
