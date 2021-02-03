import os
import sys
import platform
import subprocess


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


def try_to_import_ie(module="", libs=[]):
    """
    Check if IE python modules exists and in case of success
    environment will be set with given values.
    :param module: path to python module
    :param libs: list with paths to libraries
    """
    path_to_script = os.path.join(os.path.realpath(os.path.dirname(__file__)), 'check_ie_bindings.py')
    # We need to execute python modules checker in subprocess to avoid issue with environment
    # in case if previous import was unsuccessful it can fail further imports even if sys.path
    # will be restored to initial default values.
    status = subprocess.run([sys.executable, path_to_script, module, *libs])
    if status.returncode == 0:
        setup_env(module=module, libs=libs)
        return True
    else:
        return False


def find_ie_version():
    print("[ IMPORT ] Checking default IE Python module")
    if try_to_import_ie():
        return True

    python_version = 'python{}.{}'.format(sys.version_info[0], sys.version_info[1])

    script_path = os.path.realpath(os.path.dirname(__file__))
    bindings_paths = [
        # TODO: check that path is correct for all distribution types
        {
            "module": os.path.join(script_path, '../../../../python/', python_version),
            "libs": [
                os.path.join(script_path, '../../../inference_engine/bin/intel64/Release'),
                os.path.join(script_path, '../../../inference_engine/external/tbb/bin'),
                os.path.join(script_path, '../../../ngraph/lib'),
            ]
        },
        {
            "module": os.path.join(script_path, '../../../bin/intel64/Release/lib/python_api/', python_version)
        },
        {
            "module": os.path.join(script_path, '../../../bin/intel64/Debug/lib/python_api/', python_version)
        }
    ]

    for item in bindings_paths:
        print("[ IMPORT ] Trying to find module in {}".format(item['module']))
        if try_to_import_ie(module=item['module'], libs=item['libs'] if 'libs' in item else []):
            print("[ IMPORT ] Successfully imported IE Python modules")
            return True

    return False


if __name__ == "__main__":
    if not find_ie_version():
        exit(1)