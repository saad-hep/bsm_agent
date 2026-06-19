"""Combined SM scalar singlet basis with exact identical-field symmetry.

This module builds the full scalar invariant space for SU(2)L x SU(3)c by
forming the tensor product of the separate group singlet bases and then
projecting onto the symmetric subspace under exchanges of identical scalar
factors. This matches the logic needed for Bose symmetry in mixed
representation quartics and lower-point scalar operators.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from itertools import combinations

import sympy as sp

from .fields import FieldFactor, FieldKind
from .groups import SU2Rep, SU3Rep
from .su2_invariants import scalar_singlet_multiplicity as _su2_scalar_singlet_multiplicity
from .su2_invariants import singlet_basis as _su2_singlet_basis
from .su3_invariants import singlet_basis as _su3_singlet_basis
from .su3_invariants import singlet_multiplicity as _su3_singlet_multiplicity


@dataclass(frozen=True)
class ScalarInvariantTerm:
    coefficient: sp.Expr
    weights: tuple[int, ...]
    colors: tuple[object, ...]


def singlet_multiplicity(factors: tuple[FieldFactor, ...]) -> int:
    if not factors:
        return 0
    if any(factor.kind != FieldKind.SCALAR for factor in factors):
        raise ValueError("scalar singlet basis only supports scalar factors")
    if sum((factor.hypercharge for factor in factors), Fraction(0)) != 0:
        return 0
    if _all_color_singlets(factors):
        return _su2_scalar_singlet_multiplicity(factors)
    if _all_weak_singlets(factors):
        return _su3_singlet_multiplicity(factors, enforce_identical_scalar_symmetry=True)
    return len(singlet_basis(factors))


def singlet_basis(factors: tuple[FieldFactor, ...]) -> tuple[tuple[ScalarInvariantTerm, ...], ...]:
    if not factors:
        return ()
    if any(factor.kind != FieldKind.SCALAR for factor in factors):
        raise ValueError("scalar singlet basis only supports scalar factors")
    if sum((factor.hypercharge for factor in factors), Fraction(0)) != 0:
        return ()
    if _all_color_singlets(factors):
        return tuple(
            tuple(
                ScalarInvariantTerm(term.coefficient, term.weights, tuple(None for _ in factors))
                for term in contraction
            )
            for contraction in _su2_singlet_basis(factors, enforce_identical_scalar_symmetry=True)
        )
    if _all_weak_singlets(factors):
        return tuple(
            tuple(
                ScalarInvariantTerm(term.coefficient, tuple(0 for _ in factors), term.colors)
                for term in contraction
            )
            for contraction in _su3_singlet_basis(factors, enforce_identical_scalar_symmetry=True)
        )
    return _singlet_basis_cached(factors)


def _all_color_singlets(factors: tuple[FieldFactor, ...]) -> bool:
    return all(factor.su3 == SU3Rep(0, 0) for factor in factors)


def _all_weak_singlets(factors: tuple[FieldFactor, ...]) -> bool:
    return all(factor.su2 == SU2Rep(1) for factor in factors)


@lru_cache(maxsize=None)
def _singlet_basis_cached(factors: tuple[FieldFactor, ...]) -> tuple[tuple[ScalarInvariantTerm, ...], ...]:
    su2_basis = tuple(
        {term.weights: sp.simplify(term.coefficient) for term in contraction}
        for contraction in _su2_singlet_basis(factors, enforce_identical_scalar_symmetry=False)
    )
    su3_basis = tuple(
        {term.colors: sp.simplify(term.coefficient) for term in contraction}
        for contraction in _su3_singlet_basis(factors, enforce_identical_scalar_symmetry=False)
    )
    if not su2_basis or not su3_basis:
        return ()

    swaps = _identical_scalar_swaps(factors)
    su2_actions = {swap: _permutation_action_matrix(su2_basis, swap) for swap in swaps}
    su3_actions = {swap: _permutation_action_matrix(su3_basis, swap) for swap in swaps}

    tensor_dim = len(su2_basis) * len(su3_basis)
    if swaps:
        constraints: list[sp.Matrix] = []
        for swap in swaps:
            action = sp.kronecker_product(su2_actions[swap], su3_actions[swap])
            constraints.append(action - sp.eye(tensor_dim))
        system = constraints[0]
        for extra in constraints[1:]:
            system = system.col_join(extra)
        coefficients = [_canonicalize_vector(vector) for vector in system.nullspace()]
    else:
        coefficients = [
            tuple(sp.Integer(int(index == basis_index)) for index in range(tensor_dim))
            for basis_index in range(tensor_dim)
        ]

    unique_coefficients = _deduplicate_vectors(coefficients)
    basis_terms: list[tuple[ScalarInvariantTerm, ...]] = []
    for coefficient_vector in unique_coefficients:
        term_map: dict[tuple[tuple[int, ...], tuple[object, ...]], sp.Expr] = {}
        for tensor_index, basis_coefficient in enumerate(coefficient_vector):
            basis_coefficient = sp.simplify(basis_coefficient)
            if basis_coefficient == 0:
                continue
            su2_index, su3_index = divmod(tensor_index, len(su3_basis))
            for weights, su2_coefficient in su2_basis[su2_index].items():
                for colors, su3_coefficient in su3_basis[su3_index].items():
                    key = (weights, colors)
                    term_map[key] = sp.simplify(
                        term_map.get(key, sp.Integer(0))
                        + basis_coefficient * su2_coefficient * su3_coefficient
                    )
        terms = tuple(
            ScalarInvariantTerm(coefficient, weights, colors)
            for (weights, colors), coefficient in sorted(term_map.items(), key=lambda item: (item[0][0], tuple(map(str, item[0][1]))))
            if coefficient != 0
        )
        if terms:
            basis_terms.append(terms)
    return tuple(basis_terms)


def _identical_scalar_swaps(factors: tuple[FieldFactor, ...]) -> tuple[tuple[int, int], ...]:
    swaps = []
    for left, right in combinations(range(len(factors)), 2):
        if factors[left] == factors[right]:
            swaps.append((left, right))
    return tuple(swaps)


def _permutation_action_matrix(
    basis_vectors: tuple[dict[tuple[object, ...], sp.Expr], ...],
    swap: tuple[int, int],
) -> sp.Matrix:
    keys = sorted({key for vector in basis_vectors for key in vector}, key=str)
    basis_matrix = sp.Matrix([[vector.get(key, 0) for vector in basis_vectors] for key in keys])
    columns: list[sp.Matrix] = []
    for vector in basis_vectors:
        permuted = {_permute_tuple_key(key, swap): value for key, value in vector.items()}
        permuted_column = sp.Matrix([permuted.get(key, 0) for key in keys])
        solution = basis_matrix.gauss_jordan_solve(permuted_column)[0]
        columns.append(solution)
    return sp.Matrix.hstack(*columns)


def _permute_tuple_key(key: tuple[object, ...], swap: tuple[int, int]) -> tuple[object, ...]:
    entries = list(key)
    entries[swap[0]], entries[swap[1]] = entries[swap[1]], entries[swap[0]]
    return tuple(entries)


def _canonicalize_vector(vector: sp.Matrix | tuple[sp.Expr, ...]) -> tuple[sp.Expr, ...]:
    entries = tuple(sp.simplify(entry) for entry in vector)
    pivot = next((entry for entry in entries if entry != 0), None)
    if pivot is None:
        return tuple(sp.Integer(0) for _ in entries)
    return tuple(sp.simplify(entry / pivot) for entry in entries)


def _deduplicate_vectors(vectors: list[tuple[sp.Expr, ...]] | tuple[tuple[sp.Expr, ...], ...]) -> list[tuple[sp.Expr, ...]]:
    unique: list[tuple[sp.Expr, ...]] = []
    seen: set[tuple[sp.Expr, ...]] = set()
    for vector in vectors:
        if vector in seen:
            continue
        seen.add(vector)
        unique.append(vector)
    return sorted(unique, key=_vector_sort_key)


def _vector_sort_key(vector: tuple[sp.Expr, ...]) -> tuple[object, ...]:
    non_zero = tuple(index for index, value in enumerate(vector) if value != 0)
    return (len(non_zero), non_zero, tuple(map(str, vector)))
