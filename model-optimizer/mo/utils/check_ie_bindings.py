import os
import sys
import platform


def try_to_import_ie():
    try:
        from openvino.inference_engine import IECore
        print("[ IMPORT ]     Successfully Imported: IECore")

        from openvino.offline_transformations import ApplyMOCTransformations
        print("[ IMPORT ]     Successfully Imported: ApplyMOCTransformations")
        return True
    except ImportError as e:
        print("[ IMPORT ]     ImportError: {}".format(e))
        return False


def setup_env(module="", libs=[]):
    """
    Update sys.path and os.environ with given values.
    :param module: path to python module
    :param libs: list with paths to libraries
    """
    sys.path.append(module)
    lib_env_key, delimiter = ("PATH", ";") if platform.system() == "Windows" else ("LD_LIBRARY_PATH", ":")
    if lib_env_key not in os.environ:
        os.environ[lib_env_key] = ""
    os.environ[lib_env_key] = delimiter.join([os.environ[lib_env_key], *libs])


if __name__ == "__main__":
    setup_env(module=sys.argv[1], libs=sys.argv[2:])
    if not try_to_import_ie():
        exit(1)