"""Exact SU(2) singlet tensors with explicit dual/fundamental handling."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations

import sympy as sp
from sympy.physics.wigner import clebsch_gordan

from .fields import FieldFactor, FieldKind
from .groups import SU2Rep, su2_product_counter, su2_symmetric_power


@dataclass(frozen=True)
class SU2InvariantTerm:
    coefficient: sp.Expr
    weights: tuple[int, ...]


def singlet_multiplicity(
    factors: tuple[FieldFactor, ...],
    *,
    enforce_identical_scalar_symmetry: bool = True,
) -> int:
    return len(singlet_basis(factors, enforce_identical_scalar_symmetry=enforce_identical_scalar_symmetry))


def scalar_singlet_multiplicity(factors: tuple[FieldFactor, ...]) -> int:
    """Count SU(2) singlets for scalar factors with exact Bose symmetry.

    This is much faster than constructing the explicit singlet basis because it
    works representation-theoretically:
    - repeated identical scalar factors are first decomposed with Sym^n(rep)
    - the resulting channels are then multiplied together

    Hypercharge is handled by the caller; this function only counts SU(2)
    singlets.
    """

    if not factors:
        return 0
    if any(factor.kind != FieldKind.SCALAR for factor in factors):
        raise ValueError("scalar_singlet_multiplicity only supports scalar factors")

    grouped = _group_scalar_factors(factors)
    channels = [su2_symmetric_power(rep, count) for rep, count in grouped]
    if not channels:
        return 0

    result = channels[0]
    for channel in channels[1:]:
        next_result: Counter[SU2Rep] = Counter()
        for left_rep, left_mult in result.items():
            for right_rep, right_mult in channel.items():
                for out_rep, out_mult in su2_product_counter([left_rep, right_rep]).items():
                    next_result[out_rep] += left_mult * right_mult * out_mult
        result = next_result
    return result[SU2Rep(1)]


def singlet_basis(
    factors: tuple[FieldFactor, ...],
    *,
    enforce_identical_scalar_symmetry: bool = True,
) -> list[list[SU2InvariantTerm]]:
    if not factors:
        return []

    reps = tuple(factor.su2 for factor in factors)
    if all(rep == SU2Rep(1) for rep in reps):
        return [[SU2InvariantTerm(sp.Integer(1), tuple(0 for _ in factors))]]

    dual_flags = tuple(factor.conjugate for factor in factors)
    vectors = list(_singlet_vectors(reps, dual_flags))
    if enforce_identical_scalar_symmetry:
        vectors = _restrict_identical_scalar_symmetry(vectors, factors, reps)
    if not vectors:
        return []

    basis_weights = tuple(_basis_weights(rep, dual=factor.conjugate) for rep, factor in zip(reps, factors))
    basis: list[list[SU2InvariantTerm]] = []
    for vector in vectors:
        terms: list[SU2InvariantTerm] = []
        for flat_index, coeff in enumerate(vector):
            coeff = sp.simplify(coeff)
            if coeff == 0:
                continue
            terms.append(SU2InvariantTerm(coeff, _flat_index_to_weights(flat_index, basis_weights)))
        if terms:
            basis.append(terms)
    return basis


@lru_cache(maxsize=None)
def _singlet_vectors(reps: tuple[SU2Rep, ...], dual_flags: tuple[bool, ...]) -> tuple[tuple[sp.Expr, ...], ...]:
    if not reps:
        return ()
    basis_weights = tuple(_basis_weights(rep, dual=dual) for rep, dual in zip(reps, dual_flags))
    zero_indices = _zero_weight_indices(basis_weights, reps)
    if not zero_indices:
        return ()
    total_jp = _total_raising_generator(reps, dual_flags)
    system = total_jp.extract(range(total_jp.rows), zero_indices)
    size = total_jp.cols
    vectors = []
    for reduced_vector in system.nullspace():
        full_vector = [sp.Integer(0)] * size
        for column, basis_index in enumerate(zero_indices):
            full_vector[basis_index] = sp.simplify(reduced_vector[column])
        vectors.append(tuple(full_vector))
    return tuple(_deduplicate_vectors(_canonicalize_vector(vector) for vector in vectors if any(entry != 0 for entry in vector)))


def _restrict_identical_scalar_symmetry(
    vectors: list[tuple[sp.Expr, ...]] | tuple[tuple[sp.Expr, ...], ...],
    factors: tuple[FieldFactor, ...],
    reps: tuple[SU2Rep, ...],
) -> list[tuple[sp.Expr, ...]]:
    swaps = _identical_scalar_swaps(factors)
    if not swaps or not vectors:
        return vectors

    basis = sp.Matrix.hstack(*(sp.Matrix(vector) for vector in vectors))
    basis_weights = tuple(_basis_weights(rep, dual=factor.conjugate) for rep, factor in zip(reps, factors))
    constraints: list[sp.Matrix] = []
    for swap in swaps:
        permuted_columns = [sp.Matrix(_permute_tensor_vector(vector, reps, basis_weights, swap)) for vector in vectors]
        constraints.append(sp.Matrix.hstack(*permuted_columns) - basis)
    system = constraints[0]
    for extra in constraints[1:]:
        system = system.col_join(extra)
    combinations = system.nullspace()
    if not combinations:
        return []

    restricted = [basis * combination for combination in combinations]
    return _deduplicate_vectors(_canonicalize_vector(vector) for vector in restricted)


def _total_raising_generator(reps: tuple[SU2Rep, ...], dual_flags: tuple[bool, ...]) -> sp.Matrix:
    dims = [rep.dim for rep in reps]
    size = 1
    for dim in dims:
        size *= dim
    total_jp = sp.zeros(size)
    for index, (rep, dual) in enumerate(zip(reps, dual_flags)):
        _j3, jp, _jm = _rep_generators(rep, dual=dual)
        left_dim = 1
        right_dim = 1
        for dim in dims[:index]:
            left_dim *= dim
        for dim in dims[index + 1 :]:
            right_dim *= dim
        total_jp += sp.kronecker_product(sp.eye(left_dim), jp, sp.eye(right_dim))
    return total_jp


def _zero_weight_indices(
    basis_weights: tuple[tuple[int, ...], ...],
    reps: tuple[SU2Rep, ...],
) -> list[int]:
    zero_indices: list[int] = []
    size = 1
    for rep in reps:
        size *= rep.dim
    for flat_index in range(size):
        if sum(_flat_index_to_weights(flat_index, basis_weights)) == 0:
            zero_indices.append(flat_index)
    return zero_indices


def _flat_index_to_positions(index: int, dims: list[int]) -> tuple[int, ...]:
    positions = [0] * len(dims)
    remainder = index
    for slot in range(len(dims) - 1, -1, -1):
        positions[slot] = remainder % dims[slot]
        remainder //= dims[slot]
    return tuple(positions)


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
    entries = [sp.simplify(entry) for entry in vector]
    pivot = next((entry for entry in entries if entry != 0), None)
    if pivot is None:
        return tuple(sp.Integer(0) for _ in entries)
    return tuple(sp.simplify(entry / pivot) for entry in entries)


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


def _flat_index_to_weights(index: int, basis_weights: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
    dims = [len(weights) for weights in basis_weights]
    positions = [0] * len(dims)
    remainder = index
    for slot in range(len(dims) - 1, -1, -1):
        positions[slot] = remainder % dims[slot]
        remainder //= dims[slot]
    return tuple(basis_weights[slot][positions[slot]] for slot in range(len(dims)))


def _weights_to_positions(weights: tuple[int, ...], basis_weights: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
    return tuple(basis.index(weight) for basis, weight in zip(basis_weights, weights))


def _permute_tensor_vector(
    vector: tuple[sp.Expr, ...],
    reps: tuple[SU2Rep, ...],
    basis_weights: tuple[tuple[int, ...], ...],
    swap: tuple[int, int],
) -> tuple[sp.Expr, ...]:
    permuted = [sp.Integer(0)] * len(vector)
    for flat_index, coeff in enumerate(vector):
        if coeff == 0:
            continue
        weights = list(_flat_index_to_weights(flat_index, basis_weights))
        weights[swap[0]], weights[swap[1]] = weights[swap[1]], weights[swap[0]]
        positions = _weights_to_positions(tuple(weights), basis_weights)
        permuted[_positions_to_flat_index(positions, reps)] = coeff
    return tuple(permuted)


def _positions_to_flat_index(positions: tuple[int, ...], reps: tuple[SU2Rep, ...]) -> int:
    dims = [rep.dim for rep in reps]
    index = 0
    for pos, dim in zip(positions, dims):
        index = index * dim + pos
    return index


def _identical_scalar_swaps(factors: tuple[FieldFactor, ...]) -> list[tuple[int, int]]:
    swaps: list[tuple[int, int]] = []
    for left, right in combinations(range(len(factors)), 2):
        if (
            factors[left].kind == FieldKind.SCALAR
            and factors[right].kind == FieldKind.SCALAR
            and factors[left] == factors[right]
        ):
            swaps.append((left, right))
    return swaps


def _group_scalar_factors(factors: tuple[FieldFactor, ...]) -> tuple[tuple[SU2Rep, int], ...]:
    grouped: Counter[FieldFactor] = Counter(factors)
    return tuple(
        (factor.su2, count)
        for factor, count in sorted(grouped.items(), key=lambda item: (item[0].field.name, item[0].conjugate))
    )
