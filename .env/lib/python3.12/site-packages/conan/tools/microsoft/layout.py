import os

from conan.errors import ConanException
from conan.tools.microsoft.visual import msvc_platform_from_arch


def vs_layout(conanfile):
    """
    Initialize a layout for a typical Visual Studio project.

    :param conanfile: ``< ConanFile object >`` The current recipe object. Always use ``self``.
    """
    subproject = conanfile.folders.subproject
    conanfile.folders.source = subproject or "."
    conanfile.folders.generators = os.path.join(subproject, "conan") if subproject else "conan"
    conanfile.folders.build = subproject or "."
    conanfile.cpp.source.includedirs = ["include"]

    try:
        build_type = str(conanfile.settings.build_type)
    except ConanException:
        raise ConanException("The 'vs_layout' requires the 'build_type' setting")
    try:
        arch = str(conanfile.settings.arch)
    except ConanException:
        raise ConanException("The 'vs_layout' requires the 'arch' setting")

    if arch != "None" and arch != "x86":
        msvc_arch = msvc_platform_from_arch(arch)
        if not msvc_arch:
            raise ConanException(f"The 'vs_layout' doesn't work with the arch '{arch}'. "
                                 "Accepted architectures: 'x86', 'x86_64', 'armv7', 'armv8'")
        bindirs = os.path.join(msvc_arch, build_type)
    else:
        bindirs = build_type

    conanfile.cpp.build.libdirs = [bindirs]
    conanfile.cpp.build.bindirs = [bindirs]
