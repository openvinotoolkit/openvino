# -*- coding: utf-8 -*-
# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os.path
import sys
import errno
import subprocess  # nosec
import typing
import platform
import multiprocessing
from fnmatch import fnmatchcase
from pathlib import Path
from shutil import copyfile, rmtree
from distutils.command.build import build
from distutils.command.clean import clean
from distutils.errors import DistutilsSetupError
from distutils.file_util import copy_file
from distutils import log
from setuptools import setup, find_namespace_packages, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.build_clib import build_clib
from setuptools.command.install import install

WHEEL_LIBS_INSTALL_DIR = os.path.join("openvino", "libs")
WHEEL_LIBS_PACKAGE = "openvino.libs"
PYTHON_VERSION = f"python{sys.version_info.major}.{sys.version_info.minor}"

LIBS_DIR = "bin" if platform.system() == "Windows" else "lib"
CONFIG = "Release" if platform.system() in {"Windows", "Darwin"} else ""

machine = platform.machine()
if machine == "x86_64" or machine == "AMD64":
    ARCH = "intel64"
elif machine == "X86":
    ARCH = "ia32"
elif machine == "arm" or machine == "armv7l":
    ARCH = "arm"
elif machine == "aarch64" or machine == "arm64":
    ARCH = "arm64"

# The following variables can be defined in environment or .env file
SCRIPT_DIR = Path(__file__).resolve().parents[0]
CMAKE_BUILD_DIR = os.getenv("CMAKE_BUILD_DIR", ".")
OPENVINO_BUILD_DIR = os.getenv("OPENVINO_BUILD_DIR", CMAKE_BUILD_DIR)
OPENVINO_PYTHON_BUILD_DIR = os.getenv("OPENVINO_PYTHON_BUILD_DIR", CMAKE_BUILD_DIR)
OV_RUNTIME_LIBS_DIR = os.getenv("OV_RUNTIME_LIBS_DIR", f"runtime/{LIBS_DIR}/{ARCH}/{CONFIG}")
TBB_LIBS_DIR = os.getenv("TBB_LIBS_DIR", f"runtime/3rdparty/tbb/{LIBS_DIR}")
PUGIXML_LIBS_DIR = os.getenv("PUGIXML_LIBS_DIR", f"runtime/3rdparty/pugixml/{LIBS_DIR}")
PY_PACKAGES_DIR = os.getenv("PY_PACKAGES_DIR", f"python/{PYTHON_VERSION}")
LIBS_RPATH = "$ORIGIN" if sys.platform == "linux" else "@loader_path"

LIB_INSTALL_CFG = {
    "ie_libs": {
        "name": "core",
        "prefix": "libs.core",
        "install_dir": OV_RUNTIME_LIBS_DIR,
        "rpath": LIBS_RPATH,
        "binary_dir": OPENVINO_BUILD_DIR,
    },
    "hetero_plugin": {
        "name": "hetero",
        "prefix": "libs.core",
        "install_dir": OV_RUNTIME_LIBS_DIR,
        "binary_dir": OPENVINO_BUILD_DIR,
    },
    "gpu_plugin": {
        "name": "gpu",
        "prefix": "libs.core",
        "install_dir": OV_RUNTIME_LIBS_DIR,
        "binary_dir": OPENVINO_BUILD_DIR,
    },
    "cpu_plugin": {
        "name": "cpu",
        "prefix": "libs.core",
        "install_dir": OV_RUNTIME_LIBS_DIR,
        "binary_dir": OPENVINO_BUILD_DIR,
    },
    "multi_plugin": {
        "name": "multi",
        "prefix": "libs.core",
        "install_dir": OV_RUNTIME_LIBS_DIR,
        "binary_dir": OPENVINO_BUILD_DIR,
    },
    "batch_plugin": {
        "name": "batch",
        "prefix": "libs.core",
        "install_dir": OV_RUNTIME_LIBS_DIR,
        "binary_dir": OPENVINO_BUILD_DIR,
    },
    "tbb_libs": {
        "name": "tbb",
        "prefix": "libs.tbb",
        "install_dir": TBB_LIBS_DIR,
        "binary_dir": OPENVINO_BUILD_DIR,
    },
    "pugixml_libs": {
        "name": "pugixml",
        "prefix": "libs.pugixml",
        "install_dir": PUGIXML_LIBS_DIR,
        "binary_dir": OPENVINO_BUILD_DIR,
    },
    "ir_libs": {
        "name": "ir",
        "prefix": "libs.core",
        "install_dir": OV_RUNTIME_LIBS_DIR,
        "binary_dir": OPENVINO_BUILD_DIR,
    },
    "paddle_libs": {
        "name": "paddle",
        "prefix": "libs.core",
        "install_dir": OV_RUNTIME_LIBS_DIR,
        "binary_dir": OPENVINO_BUILD_DIR,
    },
    "onnx_libs": {
        "name": "onnx",
        "prefix": "libs.core",
        "install_dir": OV_RUNTIME_LIBS_DIR,
        "binary_dir": OPENVINO_BUILD_DIR,
    },
    # uncomment once TF FE will be used in MO
    # "tensorflow_libs": {                      # noqa: E800
    #     "name": "tensorflow",                 # noqa: E800
    #     "prefix": "libs.core",                # noqa: E800
    #     "install_dir": OV_RUNTIME_LIBS_DIR,   # noqa: E800
    #     "binary_dir": OPENVINO_BUILD_DIR,     # noqa: E800
    # },                                        # noqa: E800
}

PY_INSTALL_CFG = {
    "ie_py": {
        "name": f"pyie_{PYTHON_VERSION}",
        "prefix": "site-packages",
        "install_dir": PY_PACKAGES_DIR,
        "binary_dir": OPENVINO_PYTHON_BUILD_DIR,
    },
    "ngraph_py": {
        "name": f"pyngraph_{PYTHON_VERSION}",
        "prefix": "site-packages",
        "install_dir": PY_PACKAGES_DIR,
        "binary_dir": OPENVINO_PYTHON_BUILD_DIR,
    },
    "pyopenvino": {
        "name": f"pyopenvino_{PYTHON_VERSION}",
        "prefix": "site-packages",
        "install_dir": PY_PACKAGES_DIR,
        "binary_dir": OPENVINO_PYTHON_BUILD_DIR,
    },
}


class PrebuiltExtension(Extension):
    """Initialize Extension."""

    def __init__(self, name, sources, *args, **kwargs):
        if len(sources) != 1:
            nln = "\n"
            raise DistutilsSetupError(f"PrebuiltExtension can accept only one source, but got: {nln}{nln.join(sources)}")
        super().__init__(name, sources, *args, **kwargs)


class CustomBuild(build):
    """Custom implementation of build_clib."""

    cmake_build_types = ["Release", "Debug", "RelWithDebInfo", "MinSizeRel"]
    user_options = [
        ("config=", None, "Build configuration [{types}].".format(types="|".join(cmake_build_types))),
        ("jobs=", None, "Specifies the number of jobs to use with make."),
        ("cmake-args=", None, "Additional options to be passed to CMake."),
    ]

    def initialize_options(self):
        """Set default values for all the options that this command supports."""
        super().initialize_options()
        self.build_base = "build"
        self.config = None
        self.jobs = None
        self.cmake_args = None

    def finalize_options(self):
        """Set final values for all the options that this command supports."""
        super().finalize_options()

        if not self.config:
            if self.debug:
                self.config = "Debug"
            else:
                self.announce("Set default value for CMAKE_BUILD_TYPE = Release.", level=4)
                self.config = "Release"
        else:
            build_types = [item.lower() for item in self.cmake_build_types]
            try:
                i = build_types.index(str(self.config).lower())
                self.config = self.cmake_build_types[i]
                self.debug = True if "Debug" == self.config else False
            except ValueError:
                self.announce("Unsupported CMAKE_BUILD_TYPE value: " + self.config, level=4)
                self.announce("Supported values: {types}".format(types=", ".join(self.cmake_build_types)), level=4)
                sys.exit(1)
        if self.jobs is None and os.getenv("MAX_JOBS") is not None:
            self.jobs = os.getenv("MAX_JOBS")
        self.jobs = multiprocessing.cpu_count() if self.jobs is None else int(self.jobs)

    def run(self):
        global OPENVINO_BUILD_DIR
        self.jobs = multiprocessing.cpu_count()
        plat_specifier = ".{0}-{1}.{2}".format(self.plat_name, *sys.version_info[:2])
        self.build_temp = os.path.join(self.build_base, "temp" + plat_specifier, self.config)

        # if setup.py is directly called use CMake to build product
        if OPENVINO_BUILD_DIR == ".":
            # set path to the root of OpenVINO CMakeList file
            openvino_root_dir = Path(__file__).resolve().parents[4]
            self.announce(f"Configuring cmake project: {openvino_root_dir}", level=3)
            self.spawn(["cmake", "-S" + str(openvino_root_dir),
                                 "-B" + self.build_temp,
                                 "-DCMAKE_BUILD_TYPE={type}".format(type=self.config),
                                 "-DENABLE_PYTHON=ON"])

            self.announce("Building binaries", level=3)
            self.spawn(["cmake", "--build", self.build_temp,
                                 "--config", self.config,
                                 "-j", str(self.jobs)])
            OPENVINO_BUILD_DIR = self.build_temp
        self.run_command("build_clib")

        build.run(self)
        # Copy extra package_data content filtered by find_packages
        dst = Path(self.build_lib)
        src = Path(get_package_dir(PY_INSTALL_CFG))
        exclude = ignore_patterns("*ez_setup*", "*__pycache__*", "*.egg-info*")
        for path in src.glob("**/*"):
            if path.is_dir() or exclude(str(path)):
                continue
            path_rel = path.relative_to(src)
            (dst / path_rel.parent).mkdir(exist_ok=True, parents=True)
            copyfile(path, dst / path_rel)


class PrepareLibs(build_clib):
    """Prepare prebuilt libraries."""

    def run(self):
        self.configure(LIB_INSTALL_CFG)
        self.configure(PY_INSTALL_CFG)
        self.generate_package(get_dir_list(LIB_INSTALL_CFG))

    def configure(self, install_cfg):
        """Collect prebuilt libraries. Install them to the temp directories, set rpath."""
        for comp, comp_data in install_cfg.items():
            install_prefix = comp_data.get("prefix")
            install_dir = comp_data.get("install_dir")
            binary_dir = comp_data.get("binary_dir")
            if install_dir and not os.path.isabs(install_dir):
                install_dir = os.path.join(install_prefix, install_dir)
                self.announce(f"Installing {comp}", level=3)
                self.spawn(["cmake", "--install", binary_dir,
                                     "--prefix", install_prefix,
                                     "--config", "Release",
                                     "--strip",
                                     "--component", comp_data.get("name")])
            # set rpath if applicable
            if sys.platform != "win32" and comp_data.get("rpath"):
                for path in filter(
                    lambda x: any(item in ([".so"] if sys.platform == "linux" else [".dylib", ".so"]) for item in x.suffixes), Path(install_dir).glob("*"),
                ):
                    set_rpath(comp_data["rpath"], os.path.realpath(path))

    def generate_package(self, src_dirs):
        """Collect package data files from preinstalled dirs and put all runtime libraries to the subpackage."""
        # additional blacklist filter, just to fix cmake install issues
        blacklist = [".lib", ".pdb", "_debug.dll", "_debug.dylib"]
        package_dir = os.path.join(get_package_dir(PY_INSTALL_CFG), WHEEL_LIBS_INSTALL_DIR)

        for src_dir in src_dirs:
            local_base_dir = Path(src_dir)
            for file_path in local_base_dir.rglob("*"):
                file_name = os.path.basename(file_path)
                if file_path.is_file() and not any(file_name.endswith(ext) for ext in blacklist):
                    dst_file = os.path.join(package_dir, os.path.relpath(file_path, local_base_dir))
                    os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                    copyfile(file_path, dst_file)

        if Path(package_dir).exists():
            self.announce(f"Adding {WHEEL_LIBS_PACKAGE} package", level=3)
            packages.append(WHEEL_LIBS_PACKAGE)
            package_data.update({WHEEL_LIBS_PACKAGE: ["*"]})


class CopyExt(build_ext):
    """Copy extension files to the build directory."""

    def run(self):
        if len(self.extensions) == 1:
            self.run_command("build_clib")
            self.extensions = []
            self.extensions = find_prebuilt_extensions(get_dir_list(PY_INSTALL_CFG))
        for extension in self.extensions:
            if not isinstance(extension, PrebuiltExtension):
                raise DistutilsSetupError(f"copy_ext can accept PrebuiltExtension only, but got {extension.name}")
            src = extension.sources[0]
            dst = self.get_ext_fullpath(extension.name)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            # setting relative path to find dlls
            if sys.platform != "win32":
                rpath = os.path.relpath(get_package_dir(PY_INSTALL_CFG), os.path.dirname(src))
                if sys.platform == "linux":
                    rpath = os.path.join("$ORIGIN", rpath, WHEEL_LIBS_INSTALL_DIR)
                elif sys.platform == "darwin":
                    rpath = os.path.join("@loader_path", rpath, WHEEL_LIBS_INSTALL_DIR)
                set_rpath(rpath, os.path.realpath(src))
            copy_file(src, dst, verbose=self.verbose, dry_run=self.dry_run)


class CustomInstall(install):
    """Enable build_clib during the installation."""

    def run(self):
        self.run_command("build")
        install.run(self)


class CustomClean(clean):
    """Clean up staging directories."""

    def clean(self, install_cfg):
        for comp, comp_data in install_cfg.items():
            install_prefix = comp_data.get("prefix")
            self.announce(f"Cleaning {comp}: {install_prefix}", level=3)
            if os.path.exists(install_prefix):
                rmtree(install_prefix)

    def run(self):
        self.clean(LIB_INSTALL_CFG)
        self.clean(PY_INSTALL_CFG)
        clean.run(self)


def ignore_patterns(*patterns):
    """Filter names by given patterns."""
    return lambda name: any(fnmatchcase(name, pat=pat) for pat in patterns)


def is_tool(name):
    """Check if the command-line tool is available."""
    try:
        devnull = subprocess.DEVNULL
        subprocess.Popen([name], stdout=devnull, stderr=devnull).communicate()  # nosec
    except OSError as error:
        if error.errno == errno.ENOENT:
            return False
    return True


def remove_rpath(file_path):
    """Remove rpath from binaries.

    :param file_path: binary path
    :type file_path: pathlib.Path
    """
    if sys.platform == "darwin":
        cmd = (
            f"otool -l {file_path} "  # noqa: P103
            f"| grep LC_RPATH -A3 "
            f"| grep -o 'path.*' "
            f"| cut -d ' ' -f2 "
            f"| xargs -I{{}} install_name_tool -delete_rpath {{}} {file_path}"
        )
        if os.WEXITSTATUS(os.system(cmd)) != 0:  # nosec
            sys.exit(f"Could not remove rpath for {file_path}")
    else:
        sys.exit(f"Unsupported platform: {sys.platform}")


def set_rpath(rpath, executable):
    """Setting rpath for linux and macOS libraries."""
    print(f"Setting rpath {rpath} for {executable}")  # noqa: T001, T201
    cmd = []
    rpath_tool = ""

    if sys.platform == "linux":
        with open(os.path.realpath(executable), "rb") as file:
            if file.read(1) != b"\x7f":
                log.warn(f"WARNING: {executable}: missed ELF header")
                return
        rpath_tool = "patchelf"
        cmd = [rpath_tool, "--set-rpath", rpath, executable]
    elif sys.platform == "darwin":
        rpath_tool = "install_name_tool"
        cmd = [rpath_tool, "-add_rpath", rpath, executable]
    else:
        sys.exit(f"Unsupported platform: {sys.platform}")

    if is_tool(rpath_tool):
        if sys.platform == "darwin":
            remove_rpath(executable)
        ret_info = subprocess.run(cmd, check=True, shell=False)  # nosec
        if ret_info.returncode != 0:
            sys.exit(f"Could not set rpath: {rpath} for {executable}")
    else:
        sys.exit(f"Could not found {rpath_tool} on the system, " f"please make sure that this tool is installed")


def find_prebuilt_extensions(search_dirs):
    """Collect prebuilt python extensions."""
    extensions = []
    ext_pattern = ""
    if sys.platform == "linux":
        ext_pattern = "**/*.so"
    elif sys.platform == "win32":
        ext_pattern = "**/*.pyd"
    elif sys.platform == "darwin":
        ext_pattern = "**/*.so"
    for base_dir in search_dirs:
        for path in Path(base_dir).glob(ext_pattern):
            if path.match("openvino/libs/*"):
                continue
            relpath = path.relative_to(base_dir)
            if relpath.parent != ".":
                package_names = str(relpath.parent).split(os.path.sep)
            else:
                package_names = []
            package_names.append(path.name.split(".", 1)[0])
            name = ".".join(package_names)
            extensions.append(PrebuiltExtension(name, sources=[str(path)]))
    if not extensions:
        extensions.append(PrebuiltExtension("openvino", sources=[str("setup.py")]))
    return extensions


def get_description(desc_file_path):
    """Read description from README.md."""
    with open(desc_file_path, "r", encoding="utf-8") as fstream:
        description = fstream.read()
    return description


def get_dependencies(requirements_file_path):
    """Read dependencies from requirements.txt."""
    with open(requirements_file_path, "r", encoding="utf-8") as fstream:
        dependencies = fstream.read()
    return dependencies


def get_dir_list(install_cfg):
    """Collect all available directories with libs or python packages."""
    dirs = []
    for comp_info in install_cfg.values():
        cfg_prefix = comp_info.get("prefix")
        cfg_dir = comp_info.get("install_dir")
        if cfg_dir:
            if not os.path.isabs(cfg_dir):
                cfg_dir = os.path.join(cfg_prefix, cfg_dir)
            if cfg_dir not in dirs:
                dirs.append(cfg_dir)
    return dirs


def get_package_dir(install_cfg):
    """Get python package path based on config. All the packages should be located in one directory."""
    py_package_path = ""
    dirs = get_dir_list(install_cfg)
    if len(dirs) != 0:
        # setup.py support only one package directory, all modules should be located there
        py_package_path = dirs[0]
    return py_package_path


def concat_files(output_file, input_files):
    with open(output_file, "w", encoding="utf-8") as outfile:
        for filename in input_files:
            with open(filename, "r", encoding="utf-8") as infile:
                content = infile.read()
                outfile.write(content)
    return output_file


platforms = ["linux", "win32", "darwin"]
if not any(pl in sys.platform for pl in platforms):
    sys.exit(f"Unsupported platform: {sys.platform}, expected: linux, win32, darwin")

# copy license file into the build directory
package_license = os.getenv("WHEEL_LICENSE", SCRIPT_DIR.parents[3] / "LICENSE")
if os.path.exists(package_license):
    copyfile(package_license, "LICENSE")

packages = find_namespace_packages(get_package_dir(PY_INSTALL_CFG))
package_data: typing.Dict[str, list] = {}
pkg_name = os.getenv("WHEEL_PACKAGE_NAME", "openvino")
ext_modules = find_prebuilt_extensions(get_dir_list(PY_INSTALL_CFG)) if pkg_name == "openvino" else []

description_md = SCRIPT_DIR.parents[3] / "docs" / "install_guides" / "pypi-openvino-rt.md"
md_files = [description_md, SCRIPT_DIR.parents[3] / "docs" / "install_guides" / "pre-release-note.md"]
docs_url = "https://docs.openvino.ai/latest/index.html"

if (os.getenv("CI_BUILD_DEV_TAG")):
    output = Path.cwd() / "build" / "pypi-openvino-rt.md"
    output.parent.mkdir(exist_ok=True)
    description_md = concat_files(output, md_files)
    docs_url = "https://docs.openvino.ai/nightly/index.html"


setup(
    version=os.getenv("WHEEL_VERSION", "0.0.0"),
    build=os.getenv("WHEEL_BUILD", "000"),
    author_email=os.getenv("WHEEL_AUTHOR_EMAIL", "openvino_pushbot@intel.com"),
    name=pkg_name,
    license=os.getenv("WHEEL_LICENCE_TYPE", "OSI Approved :: Apache Software License"),
    author=os.getenv("WHEEL_AUTHOR", "Intel(R) Corporation"),
    description=os.getenv("WHEEL_DESC", "OpenVINO(TM) Runtime"),
    install_requires=get_dependencies(os.getenv("WHEEL_REQUIREMENTS", SCRIPT_DIR.parents[0] / "requirements.txt")),
    long_description=get_description(os.getenv("WHEEL_OVERVIEW", description_md)),
    long_description_content_type="text/markdown",
    download_url=os.getenv("WHEEL_DOWNLOAD_URL", "https://github.com/openvinotoolkit/openvino/tags"),
    url=os.getenv("WHEEL_URL", docs_url),
    cmdclass={
        "build": CustomBuild,
        "install": CustomInstall,
        "build_clib": PrepareLibs,
        "build_ext": CopyExt,
        "clean": CustomClean,
    },
    ext_modules=ext_modules,
    packages=packages,
    package_dir={"": get_package_dir(PY_INSTALL_CFG)},
    package_data=package_data,
    zip_safe=False,
)
