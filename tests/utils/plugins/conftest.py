"""Local pytest plugin for tests execution."""

import tempfile
from pathlib import Path

import pytest


@pytest.fixture(scope="function")
def temp_dir(pytestconfig):
    """Create temporary directory for test purposes.
    It will be cleaned up after every test run.
    """
    temp_dir = tempfile.TemporaryDirectory()
    yield Path(temp_dir.name)
    temp_dir.cleanup()
