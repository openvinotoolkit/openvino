import os
import pytest


def model_path(is_myriad=False):
    if os.environ.get("MODELS_PATH"):
        path_to_repo = os.environ.get("MODELS_PATH")
    else:
        raise EnvironmentError("MODELS_PATH variable isn't set")
    if not is_myriad:
        test_xml = os.path.join(path_to_repo, "models", "test_model", 'test_model_fp32.xml')
        test_bin = os.path.join(path_to_repo, "models", "test_model", 'test_model_fp32.bin')
    else:
        test_xml = os.path.join(path_to_repo, "models", "test_model", 'test_model_fp16.xml')
        test_bin = os.path.join(path_to_repo, "models", "test_model", 'test_model_fp16.bin')
    return (test_xml, test_bin)

def image_path():
    if os.environ.get("DATA_PATH"):
        path_to_repo = os.environ.get("DATA_PATH")
    else:
        raise EnvironmentError("DATA_PATH variable isn't set")
    path_to_img = os.path.join(path_to_repo, 'validation_set', '224x224', 'dog.bmp')
    return path_to_img

def plugins_path():
    if os.environ.get("DATA_PATH"):
        path_to_repo = os.environ.get("DATA_PATH")
    else:
        raise EnvironmentError("DATA_PATH variable isn't set")
    plugins_xml = os.path.join(path_to_repo, 'ie_class', 'plugins.xml')
    plugins_win_xml = os.path.join(path_to_repo, 'ie_class', 'plugins_mingw.xml')
    plugins_osx_xml = os.path.join(path_to_repo, 'ie_class', 'plugins_apple.xml')
    return (plugins_xml, plugins_win_xml, plugins_osx_xml)

@pytest.fixture(scope='session')
def device():
    return os.environ.get("TEST_DEVICE") if os.environ.get("TEST_DEVICE") else "CPU"
