"""Sparse tensor contraction helpers.

This module keeps tensor objects abstract and contracts them by labeled indices.
It is intentionally small and only implements the core functionality needed by
the supported quartic scalar structures.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import sympy as sp


@dataclass(frozen=True)
class SparseTensorObject:
    entries: tuple[tuple[tuple[int, ...], sp.Expr], ...]
    indices: tuple[str, ...]
    dimensions: tuple[int, ...]

    @classmethod
    def from_entries(
        cls,
        entries: dict[tuple[int, ...], sp.Expr] | list[tuple[tuple[int, ...], sp.Expr]],
        indices: tuple[str, ...],
        dimensions: tuple[int, ...],
    ) -> "SparseTensorObject":
        if isinstance(entries, dict):
            items = entries.items()
        else:
            items = entries
        normalized = []
        for position, coeff in items:
            coeff = sp.simplify(coeff)
            if coeff != 0:
                normalized.append((tuple(position), coeff))
        normalized.sort(key=lambda item: item[0])
        return cls(tuple(normalized), tuple(indices), tuple(dimensions))

    def to_dict(self) -> dict[tuple[int, ...], sp.Expr]:
        return dict(self.entries)


def contract_tensor_objects(
    objects: list[SparseTensorObject] | tuple[SparseTensorObject, ...],
    result_indices: tuple[str, ...],
) -> SparseTensorObject:
    if not objects:
        return SparseTensorObject.from_entries({(): sp.Integer(1)}, (), ())
    current = list(objects)
    while len(current) > 1:
        pair = _best_pair_to_contract(current)
        left = current.pop(pair[1])
        right = current.pop(pair[0])
        current.append(_contract_two_tensors(right, left))
    return _reorder_tensor(current[0], result_indices)


def _best_pair_to_contract(objects: list[SparseTensorObject]) -> tuple[int, int]:
    best_pair = (0, 1)
    best_score: tuple[int, int] | None = None
    for left_index in range(len(objects)):
        for right_index in range(left_index + 1, len(objects)):
            shared = set(objects[left_index].indices).intersection(objects[right_index].indices)
            result_rank = len(objects[left_index].indices) + len(objects[right_index].indices) - 2 * len(shared)
            score = (-len(shared), result_rank)
            if best_score is None or score < best_score:
                best_score = score
                best_pair = (left_index, right_index)
    return best_pair


def _contract_two_tensors(left: SparseTensorObject, right: SparseTensorObject) -> SparseTensorObject:
    shared = [index for index in left.indices if index in right.indices]
    if not shared:
        return _outer_product(left, right)

    left_positions = {label: pos for pos, label in enumerate(left.indices)}
    right_positions = {label: pos for pos, label in enumerate(right.indices)}
    left_shared = tuple(left_positions[label] for label in shared)
    right_shared = tuple(right_positions[label] for label in shared)
    left_keep = tuple(pos for pos, label in enumerate(left.indices) if label not in shared)
    right_keep = tuple(pos for pos, label in enumerate(right.indices) if label not in shared)

    left_grouped: defaultdict[tuple[int, ...], list[tuple[tuple[int, ...], sp.Expr]]] = defaultdict(list)
    for position, coeff in left.entries:
        key = tuple(position[pos] for pos in left_shared)
        left_grouped[key].append((position, coeff))

    right_grouped: defaultdict[tuple[int, ...], list[tuple[tuple[int, ...], sp.Expr]]] = defaultdict(list)
    for position, coeff in right.entries:
        key = tuple(position[pos] for pos in right_shared)
        right_grouped[key].append((position, coeff))

    result_entries: defaultdict[tuple[int, ...], sp.Expr] = defaultdict(lambda: sp.Integer(0))
    for key in set(left_grouped).intersection(right_grouped):
        for left_position, left_coeff in left_grouped[key]:
            for right_position, right_coeff in right_grouped[key]:
                result_position = tuple(left_position[pos] for pos in left_keep) + tuple(
                    right_position[pos] for pos in right_keep
                )
                result_entries[result_position] += left_coeff * right_coeff

    result_indices = tuple(label for label in left.indices if label not in shared) + tuple(
        label for label in right.indices if label not in shared
    )
    result_dimensions = tuple(left.dimensions[pos] for pos in left_keep) + tuple(right.dimensions[pos] for pos in right_keep)
    return SparseTensorObject.from_entries(result_entries, result_indices, result_dimensions)


def _outer_product(left: SparseTensorObject, right: SparseTensorObject) -> SparseTensorObject:
    result_entries: defaultdict[tuple[int, ...], sp.Expr] = defaultdict(lambda: sp.Integer(0))
    for left_position, left_coeff in left.entries:
        for right_position, right_coeff in right.entries:
            result_entries[left_position + right_position] += left_coeff * right_coeff
    return SparseTensorObject.from_entries(
        result_entries,
        left.indices + right.indices,
        left.dimensions + right.dimensions,
    )


def _reorder_tensor(tensor: SparseTensorObject, result_indices: tuple[str, ...]) -> SparseTensorObject:
    if tensor.indices == result_indices:
        return tensor
    if sorted(tensor.indices) != sorted(result_indices):
        raise ValueError("Tensor indices do not match requested result order")
    permutation = tuple(tensor.indices.index(label) for label in result_indices)
    result_entries = {
        tuple(position[index] for index in permutation): coeff
        for position, coeff in tensor.entries
    }
    result_dimensions = tuple(tensor.dimensions[index] for index in permutation)
    return SparseTensorObject.from_entries(result_entries, result_indices, result_dimensions)
