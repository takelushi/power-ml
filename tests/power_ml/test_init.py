"""Test init."""

import power_ml as ml


def test_version():
    """Test __version__."""
    assert ml.__version__ == '0.1.0'
