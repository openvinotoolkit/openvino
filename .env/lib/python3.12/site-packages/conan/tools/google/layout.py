import os


def bazel_layout(conanfile, src_folder=".", build_folder=".", target_folder=None):
    """Bazel layout is so limited. It does not allow to create its special symlinks in other
    folder. See more information in https://bazel.build/remote/output-directories"""
    subproject = conanfile.folders.subproject
    conanfile.folders.source = src_folder if not subproject else os.path.join(subproject, src_folder)
    # Bazel always builds the whole project in the root folder, but consumer can put another one
    conanfile.folders.build = build_folder if not subproject else os.path.join(subproject, build_folder)
    generators_folder = conanfile.folders.generators or "conan"
    conanfile.folders.generators = os.path.join(conanfile.folders.build, generators_folder)
    bindirs = os.path.join(conanfile.folders.build, "bazel-bin")
    libdirs = os.path.join(conanfile.folders.build, "bazel-bin")
    # Target folder is useful for working on editable mode
    if target_folder:
        bindirs = os.path.join(bindirs, target_folder)
        libdirs = os.path.join(libdirs, target_folder)
    conanfile.cpp.build.bindirs = [bindirs]
    conanfile.cpp.build.libdirs = [libdirs]
