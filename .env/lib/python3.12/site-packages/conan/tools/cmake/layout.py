import os
import tempfile

from conan.internal.graph.graph import RECIPE_CONSUMER, RECIPE_EDITABLE
from conan.errors import ConanException


def cmake_layout(conanfile, generator=None, src_folder=".", build_folder="build"):
    """
    :param conanfile: The current recipe object. Always use ``self``.
    :param generator: Allow defining the CMake generator. In most cases it doesn't need to be passed,
                      as it will get the value from the configuration
                      ``tools.cmake.cmaketoolchain:generator``, or it will automatically deduce
                      the generator from the ``settings``
    :param src_folder: Value for ``conanfile.folders.source``, change it if your source code
                       (and CMakeLists.txt) is in a subfolder.
    :param build_folder: Specify the name of the "base" build folder. The default is "build", but
                        if that folder name is used by the project, a different one can be defined
    """
    gen = conanfile.conf.get("tools.cmake.cmaketoolchain:generator", default=generator)
    if gen:
        multi = "Visual" in gen or "Xcode" in gen or "Multi-Config" in gen
    else:
        compiler = conanfile.settings.get_safe("compiler")
        if compiler == "msvc":
            multi = True
        else:
            multi = False

    subproject = conanfile.folders.subproject
    conanfile.folders.source = src_folder if not subproject else os.path.join(subproject, src_folder)
    try:
        build_type = str(conanfile.settings.build_type)
    except ConanException:
        raise ConanException("'build_type' setting not defined, it is necessary for cmake_layout()")

    try:  # TODO: Refactor this repeated pattern to deduce "is-consumer"
        if conanfile._conan_node.recipe in (RECIPE_CONSUMER, RECIPE_EDITABLE):
            folder = "test_folder" if conanfile.tested_reference_str else "build_folder"
            build_folder = conanfile.conf.get(f"tools.cmake.cmake_layout:{folder}") or build_folder
            if build_folder == "$TMP" and folder == "test_folder":
                build_folder = tempfile.mkdtemp()
    except AttributeError:
        pass

    build_folder = build_folder if not subproject else os.path.join(subproject, build_folder)
    config_build_folder, user_defined_build = get_build_folder_custom_vars(conanfile)
    if config_build_folder:
        build_folder = os.path.join(build_folder, config_build_folder)
    if not multi and not user_defined_build:
        build_folder = os.path.join(build_folder, build_type)
    conanfile.folders.build = build_folder

    conanfile.folders.generators = os.path.join(conanfile.folders.build, "generators")

    conanfile.cpp.source.includedirs = ["include"]

    if multi:
        conanfile.cpp.build.libdirs = ["{}".format(build_type)]
        conanfile.cpp.build.bindirs = ["{}".format(build_type)]
    else:
        conanfile.cpp.build.libdirs = ["."]
        conanfile.cpp.build.bindirs = ["."]


def get_build_folder_custom_vars(conanfile):
    conanfile_vars = conanfile.folders.build_folder_vars
    build_vars = conanfile.conf.get("tools.cmake.cmake_layout:build_folder_vars", check_type=list)
    if conanfile.tested_reference_str:
        if build_vars is None:  # The user can define conf build_folder_vars = [] for no vars
            build_vars = conanfile_vars or \
                     ["settings.compiler", "settings.compiler.version", "settings.arch",
                      "settings.compiler.cppstd", "settings.build_type", "options.shared"]
    else:
        try:
            is_consumer = conanfile._conan_node.recipe in (RECIPE_CONSUMER, RECIPE_EDITABLE)
        except AttributeError:
            is_consumer = False
        if is_consumer:
            if build_vars is None:
                build_vars = conanfile_vars or []
        else:
            build_vars = conanfile_vars or []

    ret = []
    for s in build_vars:
        group, var = s.split(".", 1)
        tmp = None
        if group == "settings":
            tmp = conanfile.settings.get_safe(var)
            if tmp and var == "arch":  # handle Apple multi-arch/universal binaries
                tmp = tmp.replace("|", "_")
        elif group == "options":
            value = conanfile.options.get_safe(var)
            if value is not None:
                if var == "shared":
                    tmp = "shared" if value else "static"
                else:
                    tmp = "{}_{}".format(var, value)
        elif group == "self":
            tmp = getattr(conanfile, var, None)
        elif group == "const":
            tmp = var
        else:
            raise ConanException("Invalid 'tools.cmake.cmake_layout:build_folder_vars' value, it has"
                                 f" to start with 'settings.', 'options.', 'self.' or 'const.': {s}")
        if tmp:
            ret.append(tmp.lower())

    user_defined_build = "settings.build_type" in build_vars
    return "-".join(ret), user_defined_build
