import os


def get_mo_version():
    version_txt = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir, "version.txt")
    if not os.path.isfile(version_txt):
        return "unknown version"
    with open(version_txt) as f:
        version = f.readline().replace('\n', '')
    return version


def try_to_import_ie():
    try:
        from openvino.inference_engine import IECore, get_version
        print("[ IMPORT ]     Successfully Imported: IECore")

        from openvino.offline_transformations import ApplyMOCTransformations
        print("[ IMPORT ]     Successfully Imported: ApplyMOCTransformations")

        ie_version = get_version()
        mo_version = get_mo_version()

        if mo_version in ie_version:
            print("[ IMPORT ] MO and IE versions match")
        else:
            print("[ WARNING ] MO and IE versions do no match: MO: {}, IE: {}".format(mo_version, ie_version))

        return True
    except ImportError as e:
        print("[ IMPORT ]     ImportError: {}".format(e))
        return False


if __name__ == "__main__":
    if not try_to_import_ie():
        exit(1)