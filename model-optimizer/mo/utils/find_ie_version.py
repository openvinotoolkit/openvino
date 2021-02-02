import os
import sys


def try_to_import_ie():
    try:
        from openvino.inference_engine import IECore
        print("[ IMPORT ] Successfully imported: IECore")

        from openvino.offline_transformations import ApplyMOCTransformations
        print("[ IMPORT ] Successfully imported: ApplyMOCTransformations")

        return True
    except ImportError as e:
        print("[ IMPORT ] ImportError: {}".format(e))
        return False


def find_ie_version():
    if try_to_import_ie():
        return True

    python_version = 'python{}.{}'.format(sys.version_info[0], sys.version_info[1])

    script_path = os.path.realpath(os.path.dirname(__file__))
    bindings_paths = [
        # TODO: check that path is correct for all distribution types
        os.path.join(script_path, '../../../../python/', python_version),
        os.path.join(script_path, '../../../bin/intel64/Release/lib/python_api/', python_version),
        os.path.join(script_path, '../../../bin/intel64/Debug/lib/python_api/', python_version),
    ]

    for path in bindings_paths:
        path = os.path.normpath(path)
        if not os.path.exists(path):
            print("[ IMPORT ] Path doesn't exists: {}".format(path))
            continue

        print("[ IMPORT ] Check path: {}".format(path))

        sys.path.append(path)
        if try_to_import_ie():
            print("[ IMPORT ] Bindings were found in {}".format(path))
            return True
        else:
            print("[ IMPORT ] Continue searching")
            sys.path.pop()

    return False


if __name__ == "__main__":
    if find_ie_version() == False:
        exit(1)