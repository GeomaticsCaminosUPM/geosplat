"""Pytest configuration: slow end-to-end tests are opt-in (``--run-slow``)."""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the ``--run-slow`` option.

    Args:
        parser: Pytest command-line parser.
    """
    parser.addoption("--run-slow", action="store_true", help="Run slow end-to-end tests (GPU, ~1-2 min each).")


def pytest_configure(config: pytest.Config) -> None:
    """Declare the ``slow`` marker.

    Args:
        config: Pytest configuration.
    """
    config.addinivalue_line("markers", "slow: end-to-end test, run with --run-slow")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip slow tests unless ``--run-slow`` is given.

    Args:
        config: Pytest configuration.
        items: Collected test items.
    """
    if config.getoption("--run-slow"):
        return
    skip = pytest.mark.skip(reason="slow end-to-end test: use --run-slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
