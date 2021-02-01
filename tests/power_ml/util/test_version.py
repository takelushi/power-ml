"""Test version."""

import pytest

from power_ml.util import version


def test_get_versions():
    """Test get_versions()."""
    targets = {'pytest': 'pytest'}
    expected = [('pytest', True, pytest.__version__)]
    actual = list(version.get_versions(targets))
    assert actual == expected
