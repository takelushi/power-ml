"""Test seed."""

import random

from power_ml.util import seed


def test_set_seed():
    """Test set_seed()."""
    seed_value = 4
    seed.set_seed(seed_value)
    expected = random.random()
    seed.set_seed(seed_value)
    assert random.random() == expected

    seed.set_seed(1)
    assert random.random() != expected


def test_set_seed_random():
    """Test set_seed_random()."""
    seed_value = seed.set_seed_random()
    random_value = random.random()

    new_seed_value = seed.set_seed_random()
    new_random_value = random.random()

    assert new_seed_value != seed_value
    assert new_random_value != random_value
