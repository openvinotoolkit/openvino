import json
import os
from shutil import which

from conan.tools.build import cmd_args_to_string
from conan.errors import ConanException


def vs_installation_path(version):
    return _vs_installation_path(version)[0]


def vs_detect_update(version):
    version = {"195": "18", "194": "17", "193": "17", "192": "16", "191": "15"}.get(str(version))
    full_version = _vs_installation_path(version)[1]
    components = full_version.split(".")
    if len(components) > 1:
        return components[1]


def _vs_installation_path(version):
    # TODO: Preference hardcoded, [conf] must be defined
    preference = ["Enterprise", "Professional", "Community", "BuildTools"]

    # Try with vswhere()
    try:
        legacy_products = vswhere(legacy=True)
        products = vswhere(products=["*"])
        products.extend(p for p in legacy_products if p not in products)
    except ConanException:
        products = None

    if products:  # First matching
        for product_type in preference:
            for product in products:
                if product["installationVersion"].startswith(f"{version}."):
                    if product_type in product.get("productId", ""):
                        return product["installationPath"], product["installationVersion"]

        # Append products without "productId" (Legacy installations)
        for product in products:
            if (product["installationVersion"].startswith(f"{version}.")
                    and "productId" not in product):
                return product["installationPath"], product["installationVersion"]

    # If vswhere does not find anything or not available, try with vs_comntools
    vs_path = os.getenv("vs%s0comntools" % version)
    if vs_path:
        sub_path_to_remove = os.path.join("", "Common7", "Tools", "")
        # Remove '\\Common7\\Tools\\' to get same output as vswhere
        if vs_path.endswith(sub_path_to_remove):
            vs_path = vs_path[:-(len(sub_path_to_remove)+1)]

    return vs_path, None


def vswhere(all_=False, prerelease=True, products=None, requires=None, version="", latest=False,
            legacy=False, property_="", nologo=True):

    # 'version' option only works if Visual Studio 2017 is installed:
    # https://github.com/Microsoft/vswhere/issues/91

    products = list() if products is None else products
    requires = list() if requires is None else requires

    if legacy and (products or requires):
        raise ConanException("The 'legacy' parameter cannot be specified with either the "
                             "'products' or 'requires' parameter")

    installer_path = None
    program_files = os.environ.get("ProgramFiles(x86)") or os.environ.get("ProgramFiles")
    if program_files:
        expected_path = os.path.join(program_files, "Microsoft Visual Studio", "Installer",
                                     "vswhere.exe")
        if os.path.isfile(expected_path):
            installer_path = expected_path
    vswhere_path = installer_path or which("vswhere")

    if not vswhere_path:
        raise ConanException("Cannot locate vswhere in 'Program Files'/'Program Files (x86)' "
                             "directory nor in PATH")

    arguments = list()
    arguments.append(vswhere_path)

    # Output json format
    arguments.append("-utf8")
    arguments.append("-format")
    arguments.append("json")

    if all_:
        arguments.append("-all")

    if prerelease:
        arguments.append("-prerelease")

    if products:
        arguments.append("-products")
        arguments.extend(products)

    if requires:
        arguments.append("-requires")
        arguments.extend(requires)

    if len(version) != 0:
        arguments.append("-version")
        arguments.append(version)

    if latest:
        arguments.append("-latest")

    if legacy:
        arguments.append("-legacy")

    if len(property_) != 0:
        arguments.append("-property")
        arguments.append(property_)

    if nologo:
        arguments.append("-nologo")

    try:
        from conan.internal.util.runners import check_output_runner
        cmd = cmd_args_to_string(arguments)
        output = check_output_runner(cmd).strip()
        # Ignore the "description" field, that even decoded contains non valid charsets for json
        # (ignored ones)
        output = "\n".join([line for line in output.splitlines()
                            if not line.strip().startswith('"description"')])

    except (ValueError, UnicodeDecodeError) as e:
        raise ConanException("vswhere error: %s" % str(e))

    return json.loads(output)
