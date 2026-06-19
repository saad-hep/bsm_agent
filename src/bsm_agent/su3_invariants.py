"""General SU(3) singlet construction for low-dimensional color irreps.

This module constructs invariant tensors directly from representation tensor
realizations over the fundamental SU(3) index space.  It supports the reps
needed by the package's current BSM scope:
`1, 3, bar3, 6, bar6, 8`.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations, permutations

import sympy as sp

from .fields import FieldFactor, FieldKind
from .groups import SU3Rep
from .tensor_algebra import SparseTensorObject, contract_tensor_objects

ColorLabel = int | tuple[int, int] | None


@dataclass(frozen=True)
class SU3InvariantTerm:
    coefficient: sp.Expr
    colors: tuple[ColorLabel, ...]


@dataclass(frozen=True)
class _CountingRepData:
    weights: tuple[tuple[int, int], ...]
    e12_entries: tuple[tuple[int, int, sp.Expr], ...]
    e23_entries: tuple[tuple[int, int, sp.Expr], ...]
    dim: int


def color_labels(rep: SU3Rep) -> tuple[ColorLabel, ...]:
    if rep == SU3Rep(0, 0):
        return (None,)
    if rep in (SU3Rep(1, 0), SU3Rep(0, 1)):
        return (0, 1, 2)
    if rep in (SU3Rep(2, 0), SU3Rep(0, 2)):
        return _sextet_basis()
    if rep == SU3Rep(1, 1):
        return tuple(range(1, 9))
    raise NotImplementedError(f"SU(3) invariant basis is not implemented for rep {rep}")


def generators(rep: SU3Rep) -> tuple[sp.Matrix, ...]:
    if rep == SU3Rep(1, 0):
        return _gell_mann_generators()
    if rep == SU3Rep(0, 1):
        return tuple(-matrix.conjugate() for matrix in _gell_mann_generators())
    if rep == SU3Rep(2, 0):
        return _sextet_generators()
    if rep == SU3Rep(0, 2):
        return tuple(-matrix.conjugate() for matrix in _sextet_generators())
    if rep == SU3Rep(1, 1):
        constants = _su3_f_constants()
        matrices = []
        for a in range(1, 9):
            matrix = sp.zeros(8, 8)
            for b in range(1, 9):
                for c in range(1, 9):
                    matrix[b - 1, c - 1] = -sp.I * constants.get((a, b, c), sp.Integer(0))
            matrices.append(matrix)
        return tuple(matrices)
    if rep == SU3Rep(0, 0):
        return tuple()
    raise NotImplementedError(f"SU(3) generators are not implemented for rep {rep}")


def singlet_multiplicity_for_reps(reps: tuple[SU3Rep, ...]) -> int:
    return len(_fast_singlet_vectors_for_reps(reps))


def singlet_multiplicity(
    factors: tuple[FieldFactor, ...],
    *,
    enforce_identical_scalar_symmetry: bool = True,
) -> int:
    if not factors:
        return 0
    if all(factor.su3 == SU3Rep(0, 0) for factor in factors):
        return 1
    vectors = _restricted_fast_singlet_vectors(factors) if enforce_identical_scalar_symmetry else _fast_singlet_vectors_for_reps(tuple(factor.su3 for factor in factors))
    return len(vectors)


def singlet_basis(
    factors: tuple[FieldFactor, ...],
    *,
    enforce_identical_scalar_symmetry: bool = True,
) -> list[list[SU3InvariantTerm]]:
    if not factors:
        return []
    if all(factor.su3 == SU3Rep(0, 0) for factor in factors):
        return [[SU3InvariantTerm(sp.Integer(1), tuple(None for _ in factors))]]
    if enforce_identical_scalar_symmetry:
        if _prefer_fast_explicit_basis(factors):
            return [list(contraction) for contraction in _restricted_fast_singlet_basis_terms(factors)]
        return [list(contraction) for contraction in _restricted_singlet_basis_terms(factors)]

    vectors = _candidate_invariant_vectors(tuple(factor.su3 for factor in factors))
    if not vectors:
        return []

    return [list(contraction) for contraction in _vectors_to_basis_terms(vectors, tuple(factor.su3 for factor in factors))]


@lru_cache(maxsize=None)
def _restricted_singlet_basis_terms(
    factors: tuple[FieldFactor, ...],
) -> tuple[tuple[SU3InvariantTerm, ...], ...]:
    vectors = _restricted_candidate_vectors(factors)
    return _vectors_to_basis_terms(vectors, tuple(factor.su3 for factor in factors))


@lru_cache(maxsize=None)
def _restricted_fast_singlet_basis_terms(
    factors: tuple[FieldFactor, ...],
) -> tuple[tuple[SU3InvariantTerm, ...], ...]:
    vectors = _restricted_fast_singlet_vectors(factors)
    return _vectors_to_basis_terms(vectors, tuple(factor.su3 for factor in factors))


def _vectors_to_basis_terms(
    vectors: tuple[tuple[sp.Expr, ...], ...] | list[tuple[sp.Expr, ...]],
    reps: tuple[SU3Rep, ...],
) -> tuple[tuple[SU3InvariantTerm, ...], ...]:
    if not vectors:
        return ()
    labels = tuple(color_labels(rep) for rep in reps)
    basis: list[tuple[SU3InvariantTerm, ...]] = []
    dims = [len(slot) for slot in labels]
    for vector in vectors:
        terms: list[SU3InvariantTerm] = []
        for flat_index, coeff in enumerate(vector):
            if coeff == 0:
                continue
            positions = _flat_index_to_positions(flat_index, dims)
            colors = tuple(slot[position] for slot, position in zip(labels, positions))
            terms.append(SU3InvariantTerm(coeff, colors))
        if terms:
            basis.append(tuple(terms))
    return tuple(basis)


def _prefer_fast_explicit_basis(factors: tuple[FieldFactor, ...]) -> bool:
    reps = tuple(factor.su3 for factor in factors)
    return any(rep == SU3Rep(1, 1) for rep in reps) and _product([len(color_labels(rep)) for rep in reps]) >= 512

@lru_cache(maxsize=None)
def _fast_singlet_vectors_for_reps(reps: tuple[SU3Rep, ...]) -> tuple[tuple[sp.Expr, ...], ...]:
    if not reps:
        return ()
    datas = tuple(_counting_rep_data(rep) for rep in reps)
    dims = [data.dim for data in datas]
    zero_indices = _zero_weight_indices_for_weights(tuple(data.weights for data in datas))
    if not zero_indices:
        return ()

    e12 = _raising_restriction_matrix(datas, dims, zero_indices, "e12")
    e23 = _raising_restriction_matrix(datas, dims, zero_indices, "e23")
    if e12.rows == 0 and e23.rows == 0:
        return tuple(tuple(sp.Integer(int(index == zero_index)) for index in range(_product(dims))) for zero_index in zero_indices)

    system = e12.col_join(e23)
    vectors: list[tuple[sp.Expr, ...]] = []
    for reduced_vector in system.nullspace():
        full_vector = [sp.Integer(0)] * _product(dims)
        for column, basis_index in enumerate(zero_indices):
            full_vector[basis_index] = reduced_vector[column]
        canonical = _canonicalize_vector(tuple(full_vector))
        if any(entry != 0 for entry in canonical):
            vectors.append(canonical)
    return tuple(_linearly_independent_vectors(vectors))


@lru_cache(maxsize=None)
def _restricted_fast_singlet_vectors(
    factors: tuple[FieldFactor, ...],
) -> tuple[tuple[sp.Expr, ...], ...]:
    reps = tuple(factor.su3 for factor in factors)
    vectors = _restrict_identical_scalar_symmetry(_fast_singlet_vectors_for_reps(reps), factors, reps)
    return tuple(vectors)


@lru_cache(maxsize=None)
def _candidate_invariant_vectors(reps: tuple[SU3Rep, ...]) -> tuple[tuple[sp.Expr, ...], ...]:
    dims = [len(color_labels(rep)) for rep in reps]
    label_positions = _label_position_maps(reps)
    candidates: list[tuple[sp.Expr, ...]] = []
    for component_map in _candidate_component_maps(reps):
        vector = [sp.Integer(0)] * _product(dims)
        for labels, coeff in component_map.items():
            positions = tuple(label_positions[slot][label] for slot, label in enumerate(labels))
            vector[_positions_to_flat_index(positions, dims)] = coeff
        canonical = _canonicalize_vector(tuple(vector))
        if any(entry != 0 for entry in canonical):
            candidates.append(canonical)
    return tuple(_linearly_independent_vectors(candidates))


@lru_cache(maxsize=None)
def _restricted_candidate_vectors(
    factors: tuple[FieldFactor, ...],
) -> tuple[tuple[sp.Expr, ...], ...]:
    reps = tuple(factor.su3 for factor in factors)
    vectors = _restrict_identical_scalar_symmetry(_candidate_invariant_vectors(reps), factors, reps)
    return tuple(vectors)


@lru_cache(maxsize=None)
def _candidate_component_maps(
    reps: tuple[SU3Rep, ...],
) -> tuple[dict[tuple[ColorLabel, ...], sp.Expr], ...]:
    factor_specs = tuple(_rep_index_spec(rep) for rep in reps)
    upper_indices = tuple(index for position, spec in enumerate(factor_specs) for index in _factor_slot_indices(position, spec, "up"))
    lower_indices = tuple(index for position, spec in enumerate(factor_specs) for index in _factor_slot_indices(position, spec, "down"))
    contraction_patterns = _complete_contraction_patterns(upper_indices, lower_indices)
    candidates: list[dict[tuple[ColorLabel, ...], sp.Expr]] = []
    for pattern in contraction_patterns:
        component_map = _component_map_from_pattern(reps, factor_specs, pattern)
        if component_map:
            candidates.append(component_map)
    return tuple(candidates)


def _rep_index_spec(rep: SU3Rep) -> tuple[str, ...]:
    if rep == SU3Rep(0, 0):
        return ()
    if rep == SU3Rep(1, 0):
        return ("up",)
    if rep == SU3Rep(0, 1):
        return ("down",)
    if rep == SU3Rep(2, 0):
        return ("up", "up")
    if rep == SU3Rep(0, 2):
        return ("down", "down")
    if rep == SU3Rep(1, 1):
        return ("up", "down")
    raise NotImplementedError(f"SU(3) invariant basis is not implemented for rep {rep}")


def _factor_slot_indices(position: int, spec: tuple[str, ...], orientation: str) -> tuple[str, ...]:
    indices = []
    for rank, variance in enumerate(spec):
        if variance == orientation:
            indices.append(f"{orientation[0]}{position}_{rank}")
    return tuple(indices)


@lru_cache(maxsize=None)
def _complete_contraction_patterns(
    upper_indices: tuple[str, ...],
    lower_indices: tuple[str, ...],
) -> tuple[tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str, str], ...], tuple[tuple[str, str, str], ...]], ...]:
    patterns: set[tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str, str], ...], tuple[tuple[str, str, str], ...]]] = set()

    def walk(
        uppers: tuple[str, ...],
        lowers: tuple[str, ...],
        deltas: tuple[tuple[str, str], ...],
        upper_epsilons: tuple[tuple[str, str, str], ...],
        lower_epsilons: tuple[tuple[str, str, str], ...],
    ) -> None:
        if not uppers and not lowers:
            patterns.add(
                (
                    tuple(sorted(tuple(sorted(pair)) for pair in deltas)),
                    tuple(sorted(tuple(sorted(triple)) for triple in upper_epsilons)),
                    tuple(sorted(tuple(sorted(triple)) for triple in lower_epsilons)),
                )
            )
            return
        # If the remaining tensor has balanced upper/lower index count, a
        # complete invariant basis can be generated from delta pairings alone.
        # Any epsilon-epsilon usage in this balanced case reduces to delta
        # combinations, so exploring epsilon branches only creates redundant
        # candidate tensors and slows the solver dramatically (notably for
        # octet quartics).
        if len(uppers) == len(lowers):
            first = uppers[0]
            for match_index, lower in enumerate(lowers):
                walk(
                    uppers[1:],
                    lowers[:match_index] + lowers[match_index + 1 :],
                    deltas + ((first, lower),),
                    upper_epsilons,
                    lower_epsilons,
                )
            return
        if uppers and lowers:
            first = uppers[0]
            for match_index, lower in enumerate(lowers):
                walk(
                    uppers[1:],
                    lowers[:match_index] + lowers[match_index + 1 :],
                    deltas + ((first, lower),),
                    upper_epsilons,
                    lower_epsilons,
                )
        if len(uppers) >= 3:
            first = uppers[0]
            for remainder in combinations(uppers[1:], 2):
                triple = (first, *remainder)
                rest = tuple(index for index in uppers if index not in triple)
                walk(rest, lowers, deltas, upper_epsilons + (triple,), lower_epsilons)
        if len(lowers) >= 3:
            first = lowers[0]
            for remainder in combinations(lowers[1:], 2):
                triple = (first, *remainder)
                rest = tuple(index for index in lowers if index not in triple)
                walk(uppers, rest, deltas, upper_epsilons, lower_epsilons + (triple,))

    walk(upper_indices, lower_indices, (), (), ())
    return tuple(sorted(patterns))


def _component_map_from_pattern(
    reps: tuple[SU3Rep, ...],
    factor_specs: tuple[tuple[str, ...], ...],
    pattern: tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str, str], ...], tuple[tuple[str, str, str], ...]],
) -> dict[tuple[ColorLabel, ...], sp.Expr]:
    deltas, upper_epsilons, lower_epsilons = pattern
    tensors: list[SparseTensorObject] = []
    result_indices: list[str] = []
    for position, rep in enumerate(reps):
        slot = f"s{position}"
        result_indices.append(slot)
        tensors.append(_rename_tensor_indices(_rep_tensor_object(rep), (slot, *_factor_indices(position, factor_specs[position]))))
    tensors.extend(_rename_tensor_indices(_delta_tensor_object(), pair) for pair in deltas)
    tensors.extend(_rename_tensor_indices(_epsilon_tensor_object(), triple) for triple in upper_epsilons)
    tensors.extend(_rename_tensor_indices(_epsilon_tensor_object(), triple) for triple in lower_epsilons)
    tensor = contract_tensor_objects(tensors, tuple(result_indices))
    labels = tuple(color_labels(rep) for rep in reps)
    result: dict[tuple[ColorLabel, ...], sp.Expr] = {}
    for positions, coeff in tensor.entries:
        entry = tuple(label_set[position] for label_set, position in zip(labels, positions))
        result[entry] = coeff
    return result


def _zero_weight_indices_for_weights(
    rep_weights: tuple[tuple[tuple[int, int], ...], ...]
) -> list[int]:
    dims = [len(weights) for weights in rep_weights]
    indices: list[int] = []

    def walk(slot: int, total_w3: int, total_w8: int, flat_index: int) -> None:
        if slot == len(rep_weights):
            if total_w3 == 0 and total_w8 == 0:
                indices.append(flat_index)
            return
        for state_index, (w3, w8) in enumerate(rep_weights[slot]):
            walk(slot + 1, total_w3 + w3, total_w8 + w8, flat_index * dims[slot] + state_index)

    walk(0, 0, 0, 0)
    return indices


def _raising_restriction_matrix(
    datas: tuple[_CountingRepData, ...],
    dims: list[int],
    zero_indices: list[int],
    operator_name: str,
) -> sp.Matrix:
    columns: list[dict[int, sp.Expr]] = []
    row_positions: dict[int, int] = {}
    for zero_index in zero_indices:
        positions = _flat_index_to_positions(zero_index, dims)
        image: defaultdict[int, sp.Expr] = defaultdict(lambda: sp.Integer(0))
        for slot, position in enumerate(positions):
            entries = datas[slot].e12_entries if operator_name == "e12" else datas[slot].e23_entries
            for row, col, coeff in entries:
                if col != position:
                    continue
                next_positions = list(positions)
                next_positions[slot] = row
                image[_positions_to_flat_index(tuple(next_positions), dims)] += coeff
        filtered = {row: coeff for row, coeff in image.items() if coeff != 0}
        for row in filtered:
            row_positions.setdefault(row, len(row_positions))
        columns.append(filtered)

    matrix = sp.zeros(len(row_positions), len(zero_indices))
    for column, image in enumerate(columns):
        for row, coeff in image.items():
            matrix[row_positions[row], column] = coeff
    return matrix


def _factor_indices(position: int, spec: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(f"{variance[0]}{position}_{rank}" for rank, variance in enumerate(spec))


def _rename_tensor_indices(tensor: SparseTensorObject, indices: tuple[str, ...]) -> SparseTensorObject:
    return SparseTensorObject(tensor.entries, indices, tensor.dimensions)


def _delta_tensor_object() -> SparseTensorObject:
    return SparseTensorObject.from_entries({(index, index): sp.Integer(1) for index in range(3)}, ("i", "j"), (3, 3))


@lru_cache(maxsize=None)
def _epsilon_tensor_object() -> SparseTensorObject:
    entries: dict[tuple[int, int, int], sp.Expr] = {}
    for permutation in permutations(range(3)):
        entries[permutation] = sp.Integer(_permutation_sign(permutation))
    return SparseTensorObject.from_entries(entries, ("i", "j", "k"), (3, 3, 3))


@lru_cache(maxsize=None)
def _rep_tensor_object(rep: SU3Rep) -> SparseTensorObject:
    if rep == SU3Rep(0, 0):
        return SparseTensorObject.from_entries({(0,): sp.Integer(1)}, ("r",), (1,))
    if rep == SU3Rep(1, 0):
        return SparseTensorObject.from_entries({(index, index): sp.Integer(1) for index in range(3)}, ("r", "i"), (3, 3))
    if rep == SU3Rep(0, 1):
        return SparseTensorObject.from_entries({(index, index): sp.Integer(1) for index in range(3)}, ("r", "i"), (3, 3))
    if rep == SU3Rep(2, 0) or rep == SU3Rep(0, 2):
        return _sextet_basis_tensor_object()
    if rep == SU3Rep(1, 1):
        entries: dict[tuple[int, int, int], sp.Expr] = {}
        for basis_index, matrix in enumerate(_gell_mann_generators()):
            for row in range(3):
                for col in range(3):
                    coeff = sp.simplify(matrix[row, col])
                    if coeff != 0:
                        entries[(basis_index, row, col)] = coeff
        return SparseTensorObject.from_entries(entries, ("r", "i", "j"), (8, 3, 3))
    raise NotImplementedError(f"SU(3) tensor realization is not implemented for rep {rep}")


@lru_cache(maxsize=None)
def _counting_rep_data(rep: SU3Rep) -> _CountingRepData:
    if rep == SU3Rep(0, 0):
        return _CountingRepData(weights=((0, 0),), e12_entries=(), e23_entries=(), dim=1)
    if rep == SU3Rep(1, 0):
        return _CountingRepData(
            weights=((1, 1), (-1, 1), (0, -2)),
            e12_entries=((0, 1, sp.Integer(1)),),
            e23_entries=((1, 2, sp.Integer(1)),),
            dim=3,
        )
    if rep == SU3Rep(0, 1):
        return _CountingRepData(
            weights=((-1, -1), (1, -1), (0, 2)),
            e12_entries=((1, 0, sp.Integer(-1)),),
            e23_entries=((2, 1, sp.Integer(-1)),),
            dim=3,
        )
    if rep == SU3Rep(2, 0):
        return _CountingRepData(
            weights=((2, 2), (0, 2), (1, -1), (-2, 2), (-1, -1), (0, -4)),
            e12_entries=((0, 1, sp.Integer(2)), (1, 3, sp.Integer(1)), (2, 4, sp.Integer(1))),
            e23_entries=((1, 2, sp.Integer(1)), (3, 4, sp.Integer(2)), (4, 5, sp.Integer(1))),
            dim=6,
        )
    if rep == SU3Rep(0, 2):
        return _CountingRepData(
            weights=((-2, -2), (0, -2), (-1, 1), (2, -2), (1, 1), (0, 4)),
            e12_entries=((1, 0, sp.Integer(-2)), (3, 1, sp.Integer(-1)), (4, 2, sp.Integer(-1))),
            e23_entries=((2, 1, sp.Integer(-1)), (4, 3, sp.Integer(-2)), (5, 4, sp.Integer(-1))),
            dim=6,
        )
    if rep == SU3Rep(1, 1):
        basis = _octet_counting_basis()
        weight_lookup = {
            "E12": (2, 0),
            "E13": (1, 3),
            "E21": (-2, 0),
            "E23": (-1, 3),
            "E31": (-1, -3),
            "E32": (1, -3),
            "H1": (0, 0),
            "H2": (0, 0),
        }
        e12 = _adjoint_action_entries(_fundamental_e12_matrix(), basis)
        e23 = _adjoint_action_entries(_fundamental_e23_matrix(), basis)
        return _CountingRepData(
            weights=tuple(weight_lookup[name] for name, _ in basis),
            e12_entries=e12,
            e23_entries=e23,
            dim=8,
        )

    raise NotImplementedError(f"SU(3) counting data is not implemented for rep {rep}")


def _restrict_identical_scalar_symmetry(
    vectors: list[tuple[sp.Expr, ...]] | tuple[tuple[sp.Expr, ...], ...],
    factors: tuple[FieldFactor, ...],
    reps: tuple[SU3Rep, ...],
) -> list[tuple[sp.Expr, ...]]:
    swaps = _identical_scalar_swaps(factors)
    if not swaps or not vectors:
        return list(vectors)

    dims = [len(color_labels(rep)) for rep in reps]
    basis = sp.Matrix.hstack(*(sp.Matrix(vector) for vector in vectors))
    constraints: list[sp.Matrix] = []
    for swap in swaps:
        permuted_columns = [sp.Matrix(_permute_tensor_vector(vector, dims, swap)) for vector in vectors]
        constraints.append(sp.Matrix.hstack(*permuted_columns) - basis)
    system = constraints[0]
    for extra in constraints[1:]:
        system = system.col_join(extra)
    combinations = system.nullspace()
    if not combinations:
        return []
    restricted = [basis * combination for combination in combinations]
    return _linearly_independent_vectors(_canonicalize_vector(vector) for vector in restricted)


@lru_cache(maxsize=None)
def _label_position_maps(
    reps: tuple[SU3Rep, ...],
) -> tuple[dict[ColorLabel, int], ...]:
    return tuple({label: position for position, label in enumerate(color_labels(rep))} for rep in reps)


def _adjoint_action_entries(
    generator: sp.Matrix,
    basis: tuple[tuple[str, sp.Matrix], ...],
) -> tuple[tuple[int, int, sp.Expr], ...]:
    basis_matrix = sp.Matrix.hstack(*[_flatten_matrix(matrix) for _name, matrix in basis])
    entries: list[tuple[int, int, sp.Expr]] = []
    for col, (_name, matrix) in enumerate(basis):
        commutator = generator * matrix - matrix * generator
        coeffs = basis_matrix.LUsolve(_flatten_matrix(commutator))
        for row in range(coeffs.rows):
            coeff = sp.simplify(coeffs[row, 0])
            if coeff != 0:
                entries.append((row, col, coeff))
    return tuple(entries)


def _flatten_matrix(matrix: sp.Matrix) -> sp.Matrix:
    return sp.Matrix(matrix.rows * matrix.cols, 1, [matrix[row, col] for row in range(matrix.rows) for col in range(matrix.cols)])


@lru_cache(maxsize=None)
def _octet_counting_basis() -> tuple[tuple[str, sp.Matrix], ...]:
    h1 = sp.Matrix([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
    h2 = sp.Matrix([[0, 0, 0], [0, 1, 0], [0, 0, -1]])
    basis = (
        ("E12", sp.Matrix([[0, 1, 0], [0, 0, 0], [0, 0, 0]])),
        ("E13", sp.Matrix([[0, 0, 1], [0, 0, 0], [0, 0, 0]])),
        ("E21", sp.Matrix([[0, 0, 0], [1, 0, 0], [0, 0, 0]])),
        ("E23", sp.Matrix([[0, 0, 0], [0, 0, 1], [0, 0, 0]])),
        ("E31", sp.Matrix([[0, 0, 0], [0, 0, 0], [1, 0, 0]])),
        ("E32", sp.Matrix([[0, 0, 0], [0, 0, 0], [0, 1, 0]])),
        ("H1", h1),
        ("H2", h2),
    )
    return basis


def _fundamental_e12_matrix() -> sp.Matrix:
    return sp.Matrix([[0, 1, 0], [0, 0, 0], [0, 0, 0]])


def _fundamental_e23_matrix() -> sp.Matrix:
    return sp.Matrix([[0, 0, 0], [0, 0, 1], [0, 0, 0]])


def _permute_tensor_vector(
    vector: tuple[sp.Expr, ...],
    dims: list[int],
    swap: tuple[int, int],
) -> tuple[sp.Expr, ...]:
    permuted = [sp.Integer(0)] * len(vector)
    for flat_index, coeff in enumerate(vector):
        if coeff == 0:
            continue
        positions = list(_flat_index_to_positions(flat_index, dims))
        positions[swap[0]], positions[swap[1]] = positions[swap[1]], positions[swap[0]]
        permuted[_positions_to_flat_index(tuple(positions), dims)] = coeff
    return tuple(permuted)


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


def _positions_to_flat_index(positions: tuple[int, ...], dims: list[int]) -> int:
    index = 0
    for position, dim in zip(positions, dims):
        index = index * dim + position
    return index


def _flat_index_to_positions(index: int, dims: list[int]) -> tuple[int, ...]:
    positions = [0] * len(dims)
    remainder = index
    for slot in range(len(dims) - 1, -1, -1):
        positions[slot] = remainder % dims[slot]
        remainder //= dims[slot]
    return tuple(positions)


def _canonicalize_vector(vector: sp.Matrix | tuple[sp.Expr, ...]) -> tuple[sp.Expr, ...]:
    entries = [sp.simplify(entry) for entry in vector]
    pivot = next((entry for entry in entries if entry != 0), None)
    if pivot is None:
        return tuple(sp.Integer(0) for _ in entries)
    return tuple(sp.simplify(entry / pivot) for entry in entries)


def _linearly_independent_vectors(vectors) -> list[tuple[sp.Expr, ...]]:
    independent: list[tuple[sp.Expr, ...]] = []
    matrix = sp.zeros(0, 0)
    current_rank = 0
    for vector in sorted(vectors, key=_vector_sort_key):
        column = sp.Matrix(vector)
        trial = column if matrix.rows == 0 else sp.Matrix.hstack(matrix, column)
        trial_rank = trial.rank()
        if trial_rank > current_rank:
            matrix = trial
            current_rank = trial_rank
            independent.append(tuple(sp.simplify(entry) for entry in vector))
    return independent


def _vector_sort_key(vector: tuple[sp.Expr, ...]) -> tuple:
    non_zero = tuple(index for index, value in enumerate(vector) if value != 0)
    return (len(non_zero), non_zero, tuple(str(value) for value in vector))


def _product(values: list[int]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def _permutation_sign(permutation: tuple[int, ...]) -> int:
    sign = 1
    values = list(permutation)
    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            if values[left] > values[right]:
                sign *= -1
    return sign


def _sextet_basis() -> tuple[tuple[int, int], ...]:
    return ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))


def _sextet_pair_coeff(first: int, second: int, basis: tuple[int, int]) -> sp.Expr:
    i, j = basis
    if i == j:
        return sp.Integer(1) if first == i and second == j else sp.Integer(0)
    return sp.sqrt(2) / 2 if (first, second) in ((i, j), (j, i)) else sp.Integer(0)


@lru_cache(maxsize=None)
def _sextet_basis_tensor_object() -> SparseTensorObject:
    entries: dict[tuple[int, int, int], sp.Expr] = {}
    for basis_index, basis in enumerate(_sextet_basis()):
        for first in range(3):
            for second in range(3):
                coeff = _sextet_pair_coeff(first, second, basis)
                if coeff != 0:
                    entries[(basis_index, first, second)] = coeff
    return SparseTensorObject.from_entries(entries, ("r", "i", "j"), (6, 3, 3))


def _gell_mann_generators() -> tuple[sp.Matrix, ...]:
    i = sp.I
    matrices = (
        sp.Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 0]]),
        sp.Matrix([[0, -i, 0], [i, 0, 0], [0, 0, 0]]),
        sp.Matrix([[1, 0, 0], [0, -1, 0], [0, 0, 0]]),
        sp.Matrix([[0, 0, 1], [0, 0, 0], [1, 0, 0]]),
        sp.Matrix([[0, 0, -i], [0, 0, 0], [i, 0, 0]]),
        sp.Matrix([[0, 0, 0], [0, 0, 1], [0, 1, 0]]),
        sp.Matrix([[0, 0, 0], [0, 0, -i], [0, i, 0]]),
        sp.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, -2]]) / sp.sqrt(3),
    )
    return tuple(matrix / 2 for matrix in matrices)


def _sextet_generators() -> tuple[sp.Matrix, ...]:
    basis = _sextet_basis()
    fundamental_generators = _gell_mann_generators()
    matrices = []
    for generator in fundamental_generators:
        matrix = sp.zeros(6, 6)
        for col, in_basis in enumerate(basis):
            for row, out_basis in enumerate(basis):
                coeff = sp.Integer(0)
                for a in range(3):
                    for b in range(3):
                        in_coeff = _sextet_pair_coeff(a, b, in_basis)
                        if in_coeff == 0:
                            continue
                        for c in range(3):
                            coeff += _sextet_pair_coeff(c, b, out_basis) * generator[c, a] * in_coeff
                            coeff += _sextet_pair_coeff(a, c, out_basis) * generator[c, b] * in_coeff
                matrix[row, col] = sp.simplify(coeff)
        matrices.append(matrix)
    return tuple(matrices)


@lru_cache(maxsize=None)
def _su3_f_constants() -> dict[tuple[int, int, int], sp.Expr]:
    generators = _gell_mann_generators()
    constants: dict[tuple[int, int, int], sp.Expr] = {}
    for a, ta in enumerate(generators, start=1):
        for b, tb in enumerate(generators, start=1):
            for c, tc in enumerate(generators, start=1):
                coeff = sp.simplify(-2 * sp.I * sp.trace((ta * tb - tb * ta) * tc))
                if coeff != 0:
                    constants[(a, b, c)] = coeff
    return constants
