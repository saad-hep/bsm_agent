"""Fast indexed/tensor LaTeX for supported operators.

This module provides a symbolic tensor-basis presentation that avoids the
component explosion of full expanded output.  It is the preferred comparison
layer for large representations.
"""

from __future__ import annotations

from .expansion import _scalar_contraction_count, _su2_contraction_count, _su3_contraction_count
from .fields import FieldFactor, field_latex_name
from .groups import SU3Rep
from .operators import Operator, latex_identifier


def indexed_operator_latex(operator: Operator) -> str:
    special = _special_indexed_latex(operator)
    if special is not None:
        return special

    if operator.category == "scalar_potential":
        scalar_count = _safe_scalar_contraction_count(operator)
        tensor = "" if scalar_count == 1 else rf"\mathcal{{I}}^{{({operator.contraction_index})}}_{{\mathrm{{SM}}}} \, "
        body = r" ".join(_field_tensor_latex(factor, slot) for slot, factor in enumerate(operator.factors, start=1))
        suffix = r" + \mathrm{h.c.}" if operator.add_hc else ""
        return rf"{latex_identifier(operator.coefficient)} \, {tensor}{body}{suffix}"

    su2_count = _safe_su2_contraction_count(operator)
    su3_count = _safe_su3_contraction_count(operator)
    su2_index = ((operator.contraction_index - 1) % su2_count) + 1
    su3_index = ((operator.contraction_index - 1) // su2_count) + 1

    su3_tensor = "" if su3_count == 1 else rf"\mathcal{{I}}^{{({su3_index})}}_{{\mathrm{{SU(3)}}}} \, "
    su2_tensor = "" if su2_count == 1 else rf"\mathcal{{I}}^{{({su2_index})}}_{{\mathrm{{SU(2)}}}} \, "
    body = r" ".join(_field_tensor_latex(factor, slot) for slot, factor in enumerate(operator.factors, start=1))
    suffix = r" + \mathrm{h.c.}" if operator.add_hc else ""
    return rf"{latex_identifier(operator.coefficient)} \, {su3_tensor}{su2_tensor}{body}{suffix}"


def _safe_su2_contraction_count(operator: Operator) -> int:
    try:
        return _su2_contraction_count(tuple(operator.factors))
    except Exception:
        return 1


def _safe_su3_contraction_count(operator: Operator) -> int:
    try:
        return _su3_contraction_count(tuple(operator.factors))
    except Exception:
        return 1


def _safe_scalar_contraction_count(operator: Operator) -> int:
    try:
        return _scalar_contraction_count(tuple(operator.factors))
    except Exception:
        return 1


def _field_tensor_latex(factor: FieldFactor, slot: int) -> str:
    base = field_latex_name(factor.field.name)
    if factor.conjugate:
        base = rf"{base}^\dagger"

    su3_suffix = _su3_index_suffix(factor, slot)
    su2_suffix = _su2_index_suffix(factor, slot)
    return base + su3_suffix + su2_suffix


def _su2_index_suffix(factor: FieldFactor, slot: int) -> str:
    rank = factor.su2.dim - 1
    if rank <= 0:
        return ""
    labels = [rf"i_{{{slot}{position}}}" for position in range(1, rank + 1)]
    joined = " ".join(labels)
    if factor.conjugate:
        return rf"^{{({joined})}}"
    return rf"_{{({joined})}}"


def _su3_index_suffix(factor: FieldFactor, slot: int) -> str:
    rep = factor.su3
    if rep == SU3Rep(0, 0):
        return ""
    if rep == SU3Rep(1, 0):
        return rf"^{{a_{{{slot}}}}}"
    if rep == SU3Rep(0, 1):
        return rf"_{{a_{{{slot}}}}}"
    if rep == SU3Rep(2, 0):
        return rf"^{{(a_{{{slot}}} b_{{{slot}}})}}"
    if rep == SU3Rep(0, 2):
        return rf"_{{(a_{{{slot}}} b_{{{slot}}})}}"
    if rep == SU3Rep(1, 1):
        return rf"^{{A_{{{slot}}}}}"
    return ""


def _special_indexed_latex(operator: Operator) -> str | None:
    sextet_self = _sextet_self_special_latex(operator)
    if sextet_self is not None:
        return sextet_self
    mixed_sextet = _mixed_sextet_special_latex(operator)
    if mixed_sextet is not None:
        return mixed_sextet
    return None


def _sextet_self_special_latex(operator: Operator) -> str | None:
    if operator.category != "scalar_potential":
        return None

    names = tuple(factor.field.name for factor in operator.factors)
    if len(set(names)) != 1 or len(names) != 4:
        return None
    if tuple(factor.conjugate for factor in operator.factors) != (False, False, True, True):
        return None

    reps = tuple((factor.field.su3, factor.field.su2.dim) for factor in operator.factors)
    if not all(rep == (SU3Rep(2, 0), 1) or rep == (SU3Rep(0, 2), 1) for rep in reps):
        return None

    name = names[0]
    if operator.contraction_index == 1:
        form = rf"{name}^\dagger_{{ij}} {name}_{{ij}} {name}^\dagger_{{kl}} {name}_{{kl}}"
        return rf"{latex_identifier(operator.coefficient)} \, {form}"
    if operator.contraction_index == 2:
        form = rf"{name}^\dagger_{{ij}} {name}_{{jk}} {name}^\dagger_{{kl}} {name}_{{li}}"
        return rf"{latex_identifier(operator.coefficient)} \, {form}"
    return None


def _mixed_sextet_special_latex(operator: Operator) -> str | None:
    if operator.category != "scalar_potential":
        return None

    names = tuple(factor.field.name for factor in operator.factors)
    if len(names) != 4 or len(set(names)) != 2:
        return None
    if tuple(factor.conjugate for factor in operator.factors) != (False, True, False, True):
        return None

    reps = tuple((factor.field.su3, factor.field.su2.dim) for factor in operator.factors)
    if not all(rep == (SU3Rep(2, 0), 1) or rep == (SU3Rep(0, 2), 1) for rep in reps):
        return None

    first_name, second_name = names[0], names[2]
    if names[1] != first_name or names[3] != second_name or first_name == second_name:
        return None

    if operator.contraction_index == 1:
        form = rf"{first_name}^\dagger_{{ij}} {first_name}_{{ij}} {second_name}^\dagger_{{kl}} {second_name}_{{kl}}"
        return rf"{latex_identifier(operator.coefficient)} \, {form}"
    if operator.contraction_index == 2:
        form = rf"{first_name}^\dagger_{{ij}} {second_name}_{{ij}} {second_name}^\dagger_{{kl}} {first_name}_{{kl}}"
        return rf"{latex_identifier(operator.coefficient)} \, {form}"
    if operator.contraction_index == 3:
        form = rf"{first_name}^\dagger_{{ij}} {first_name}_{{jk}} {second_name}^\dagger_{{kl}} {second_name}_{{li}}"
        return rf"{latex_identifier(operator.coefficient)} \, {form}"
    return None
