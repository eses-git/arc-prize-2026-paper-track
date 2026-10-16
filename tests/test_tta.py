"""Tests for Test-Time Augmentation (TTA) symmetry transforms.

These verify that the 8 dihedral transformations (4 rotations x 2 flips) are
generated correctly and that the reverse transform recovers the original grid
for every transform type.
"""

from __future__ import annotations

import pytest

from tta_transforms import (
    TRANSFORM_TYPES,
    apply_tta_transform,
    generate_tta_variants,
    reverse_tta_transform,
)

_EXPECTED_TYPES = {
    "identity",
    "rot90",
    "rot180",
    "rot270",
    "fliplr",
    "fliplr_rot90",
    "fliplr_rot180",
    "fliplr_rot270",
}


def test_transform_types_are_the_8_dihedral_symmetries() -> None:
    """TRANSFORM_TYPES contains exactly the 8 expected dihedral symmetries."""
    assert set(TRANSFORM_TYPES) == _EXPECTED_TYPES
    assert len(TRANSFORM_TYPES) == 8


def test_generate_tta_variants_returns_8_variants() -> None:
    """Eight variants are produced, each preserving the pair count."""
    pairs = [
        {"input": [[1, 2], [3, 4]], "output": [[1, 0], [0, 1]]},
        {"input": [[5]], "output": [[5]]},
    ]
    variants = generate_tta_variants(pairs)

    assert len(variants) == 8
    for variant in variants:
        assert variant["transform_type"] in _EXPECTED_TYPES
        assert len(variant["pairs"]) == len(pairs)


def test_identity_variant_is_unchanged() -> None:
    """The identity variant leaves every grid untouched."""
    pairs = [{"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]}]
    variants = generate_tta_variants(pairs)
    identity = next(v for v in variants if v["transform_type"] == "identity")

    assert identity["pairs"][0]["input"] == [[1, 2], [3, 4]]
    assert identity["pairs"][0]["output"] == [[4, 3], [2, 1]]


def test_rotation_and_flip_values_match_expected() -> None:
    """Explicit expected values for the rotation and flip transforms."""
    pairs = [{"input": [[1, 2], [3, 4]], "output": [[1, 2], [3, 4]]}]
    variants = {v["transform_type"]: v for v in generate_tta_variants(pairs)}
    inputs = {t: variants[t]["pairs"][0]["input"] for t in variants}

    assert inputs["rot90"] == [[2, 4], [1, 3]]
    assert inputs["rot180"] == [[4, 3], [2, 1]]
    assert inputs["rot270"] == [[3, 1], [4, 2]]
    assert inputs["fliplr"] == [[2, 1], [4, 3]]


@pytest.mark.parametrize("transform_type", TRANSFORM_TYPES)
def test_reverse_recovers_original_for_every_transform(transform_type: str) -> None:
    """Forward then reverse returns the original grid for all 8 transforms."""
    grid = [[1, 2, 3], [4, 5, 6]]  # non-square, to exercise dimension swaps
    variants = generate_tta_variants([{"input": grid, "output": grid}])
    variant = next(v for v in variants if v["transform_type"] == transform_type)
    transformed = variant["pairs"][0]["input"]

    assert reverse_tta_transform(transformed, transform_type) == grid


def test_reverse_tta_transform_rejects_unknown_type() -> None:
    """An unknown transform type raises a clear error."""
    with pytest.raises(ValueError):
        reverse_tta_transform([[1]], "not_a_transform")


def test_apply_tta_transform_matches_generate_variants() -> None:
    """The forward transform matches the variants produced for a single pair."""
    grid = [[1, 2], [3, 4]]
    variants = generate_tta_variants([{"input": grid, "output": grid}])

    for variant in variants:
        transform_type = variant["transform_type"]
        assert apply_tta_transform(grid, transform_type) == variant["pairs"][0]["input"]
