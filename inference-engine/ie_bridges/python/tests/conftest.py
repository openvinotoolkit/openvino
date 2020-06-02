import pathlib
import os
import pytest

test_root = pathlib.Path(__file__).parent


@pytest.fixture(scope='session')
def models_dir():
    return test_root / 'test_data' / 'models'


@pytest.fixture(scope='session')
def images_dir():
    return test_root / 'test_data' / 'images'


@pytest.fixture(scope='session')
def device():
    return os.environ.get("TEST_DEVICE") if os.environ.get("TEST_DEVICE") else "CPU"
