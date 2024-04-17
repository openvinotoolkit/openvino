# -*- coding: utf-8 -*-
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os.path
import sys
import errno
import subprocess  # nosec
import typing
import platform
import re
import shutil
import multiprocessing
import logging as log
from fnmatch import fnmatchcase
from pathlib import Path
from shutil import copyfile, rmtree
from setuptools import setup, find_namespace_packages, Extension, Command
from setuptools.command.build_ext import build_ext
from setuptools.command.build_clib import build_clib
from setuptools.command.install import install
from setuptools.command.build import build
from setuptools.errors import SetupError


WHEEL_LIBS_INSTALL_DIR = os.path.join("openvino", "libs")
WHEEL_LIBS_PACKAGE = "openvino.libs"
PYTHON_VERSION = f"python{sys.version_info.major}.{sys.version_info.minor}"

LIBS_DIR = "bin" if platform.system() == "Windows" else "lib"

machine = platform.machine()
if machine == "x86_64" or machine == "AMD64":
    ARCH = "intel64"
elif machine == "X86" or machine == "i686":
    ARCH = "ia32"
elif machine == "arm" or machine == "armv7l":
    ARCH = "arm"
elif machine == "aarch64" or machine == "arm64" or machine == "ARM64":
    ARCH = "arm64"
elif machine == "riscv64":
    ARCH = "riscv64"

# The following variables can be defined in environment or .env file
SCRIPT_DIR = Path(__file__).resolve().parents[0]
WORKING_DIR = Path.cwd()
BUILD_BASE = f"{WORKING_DIR}/build_{PYTHON_VERSION}"
OPENVINO_SOURCE_DIR = SCRIPT_DIR.parents[3]
OPENVINO_BINARY_DIR = os.getenv("OPENVINO_BINARY_DIR")
OPENVINO_PYTHON_BINARY_DIR = os.getenv("OPENVINO_PYTHON_BINARY_DIR", "python_build")
CONFIG = os.getenv("BUILD_TYPE", "Release")
OV_RUNTIME_LIBS_DIR = os.getenv("OV_RUNTIME_LIBS_DIR", f"runtime/{LIBS_DIR}/{ARCH}/{CONFIG}")
TBB_LIBS_DIR = os.getenv("TBB_LIBS_DIR", f"runtime/3rdparty/tbb/{LIBS_DIR}")
PUGIXML_LIBS_DIR = os.getenv("PUGIXML_LIBS_DIR", f"runtime/3rdparty/pugixml/{LIBS_DIR}")
PY_PACKAGES_DIR = os.getenv("PY_PACKAGES_DIR", "python")
LIBS_RPATH = "$ORIGIN" if sys.platform == "linux" else "@loader_path"
PYTHON_EXTENSIONS_ONLY = True if os.getenv("PYTHON_EXTENSIONS_ONLY") is not None else False
SKIP_RPATH = True if os.getenv("SKIP_RPATH") is not None else False
CPACK_GENERATOR = os.getenv("CPACK_GENERATOR", "TGZ")

LIB_INSTALL_CFG = {
    "ie_libs": {
        "name": "core",
        "prefix": f"{BUILD_BASE}/libs.core",
        "install_dir": OV_RUNTIME_LIBS_DIR,
        "rpath": LIBS_RPATH,
        "binary_dir": OPENVINO_BINARY_DIR,
    },
    "hetero_plugin": {
        "name": "hetero",
        "prefix": f"{BUILD_BASE}/libs.hetero",
        "install_dir": OV_RUNTIME_LIBS_DIR,
        "rpath": LIBS_RPATH,
        "binary_dir": OPENVINO_BINARY_DIR,
    },
    "gpu_plugin": {
        "name": "gpu",
        "prefix": f"{BUILD_BASE}/libs.gpu",
        "install_dir": OV_RUNTIME_LIBS_DIR,
        "rpath": LIBS_RPATH,
        "binary_dir": OPENVINO_BINARY_DIR,
    },
    "cpu_plugin": {
        "name": "cpu",
        "prefix": f"{BUILD_BASE}/libs.cpu",
        "install_dir": OV_RUNTIME_LIBS_DIR,
        "rpath": LIBS_RPATH,
        "binary_dir": OPENVINO_BINARY_DIR,
    },
    "npu_plugin": {
        "name": "npu",
        "prefix": f"{BUILD_BASE}/libs.npu",
        "install_dir": OV_RUNTIME_LIBS_DIR,
        "rpath": LIBS_RPATH,
        "binary_dir": OPENVINO_BINARY_DIR,
    },
    "multi_plugin": {
        "name": "multi",
        "prefix": f"{BUILD_BASE}/libs.multi",
        "install_dir": OV_RUNTIME_LIBS_DIR,
        "rpath": LIBS_RPATH,
        "binary_dir": OPENVINO_BINARY_DIR,
    },
    "batch_plugin": {
        "name": "batch",
        "prefix": f"{BUILD_BASE}/libs.batch",
        "install_dir": OV_RUNTIME_LIBS_DIR,
        "rpath": LIBS_RPATH,
        "binary_dir": OPENVINO_BINARY_DIR,
    },
    "tbb_libs": {
        "name": "tbb",
        "prefix": f"{BUILD_BASE}/libs.tbb",
        "install_dir": TBB_LIBS_DIR,
        "rpath": LIBS_RPATH,
        "binary_dir": OPENVINO_BINARY_DIR,
    },
    "pugixml_libs": {
        "name": "pugixml",
        "prefix": f"{BUILD_BASE}/libs.pugixml",
        "install_dir": PUGIXML_LIBS_DIR,
        "binary_dir": OPENVINO_BINARY_DIR,
    },
    "ir_libs": {
        "name": "ir",
        "prefix": f"{BUILD_BASE}/libs.ir",
        "install_dir": OV_RUNTIME_LIBS_DIR,
        "rpath": LIBS_RPATH,
        "binary_dir": OPENVINO_BINARY_DIR,
    },
    "paddle_libs": {
        "name": "paddle",
        "prefix": f"{BUILD_BASE}/libs.paddle",
        "install_dir": OV_RUNTIME_LIBS_DIR,
        "rpath": LIBS_RPATH,
        "binary_dir": OPENVINO_BINARY_DIR,
    },
    "pytorch_libs": {
        "name": "pytorch",
        "prefix": f"{BUILD_BASE}/libs.pytorch",
        "install_dir": OV_RUNTIME_LIBS_DIR,
        "rpath": LIBS_RPATH,
        "binary_dir": OPENVINO_BINARY_DIR,
    },
    "onnx_libs": {
        "name": "onnx",
        "prefix": f"{BUILD_BASE}/libs.onnx",
        "install_dir": OV_RUNTIME_LIBS_DIR,
        "rpath": LIBS_RPATH,
        "binary_dir": OPENVINO_BINARY_DIR,
    },
    "tensorflow_libs": {
        "name": "tensorflow",
        "prefix": f"{BUILD_BASE}/libs.tensorflow",
        "install_dir": OV_RUNTIME_LIBS_DIR,
        "rpath": LIBS_RPATH,
        "binary_dir": OPENVINO_BINARY_DIR,
    },
    "tensorflow_lite_libs": {
        "name": "tensorflow_lite",
        "prefix": f"{BUILD_BASE}/libs.tensorflow_lite",
        "install_dir": OV_RUNTIME_LIBS_DIR,
        "rpath": LIBS_RPATH,
        "binary_dir": OPENVINO_BINARY_DIR,
    },
}

PY_INSTALL_CFG = {
    "pyopenvino": {
        "name": f"pyopenvino_{PYTHON_VERSION}",
        "prefix": f"{BUILD_BASE}/site-packages",
        "source_dir": f"{OPENVINO_SOURCE_DIR}/src/bindings/python",
        "install_dir": PY_PACKAGES_DIR,
        "binary_dir": OPENVINO_PYTHON_BINARY_DIR,
    },
    "ovc": {
        "entry_point": {
            "console_scripts": [
                "ovc = openvino.tools.ovc.main:main",
            ],
        },
        "name": "ovc",
        "prefix": f"{BUILD_BASE}/site-packages",
        "source_dir": f"{OPENVINO_SOURCE_DIR}/tools/ovc",
        "install_dir": PY_PACKAGES_DIR,
        "binary_dir": "ovc",
    },
    "benchmark_app": {
        "entry_point": {
            "console_scripts": [
                "benchmark_app = openvino.tools.benchmark.main:main",
            ],
        },
        "name": "benchmark_app",
        "prefix": f"{BUILD_BASE}/site-packages",
        "source_dir": f"{OPENVINO_SOURCE_DIR}/tools/benchmark_tool",
        "install_dir": PY_PACKAGES_DIR,
        "binary_dir": "benchmark_app",
    },
}


class PrebuiltExtension(Extension):
    """Initialize Extension."""

    def __init__(self, name, sources, *args, **kwargs):
        if len(sources) != 1:
            nln = "\n"
            raise SetupError(f"PrebuiltExtension can accept only one source, but got: {nln}{nln.join(sources)}")
        super().__init__(name, sources, *args, **kwargs)
        self._needs_stub = False


class CustomBuild(build):
    """Custom implementation of build_clib."""

    user_options = build.user_options + [
        ("jobs=", None, "Specifies the number of jobs to use with make."),
        ("cmake-args=", None, "Additional options to be passed to CMake."),
    ]

    def initialize_options(self):
        """Set default values for all the options that this command supports."""
        super().initialize_options()
        self.build_base = BUILD_BASE
        self.jobs = None
        self.cmake_args = None

    def finalize_options(self):
        """Set final values for all the options that this command supports."""
        super().finalize_options()

        if self.jobs is None and os.getenv("MAX_JOBS") is not None:
            self.jobs = os.getenv("MAX_JOBS")
        self.jobs = multiprocessing.cpu_count() if self.jobs is None else int(self.jobs)

        if self.cmake_args is None:
            self.cmake_args = ""

    def cmake_build_and_install(self, install_cfg):
        """Runs cmake (configure, build and install) if artfiacts are not already built / installed."""
        plat_specifier = ".{0}-{1}.{2}".format(self.plat_name, *sys.version_info[:2])
        self.build_temp = os.path.join(self.build_base, "temp" + plat_specifier)
        self.announce(f"Create build directory: {self.build_temp}", level=3)

        # build some components which have not been built yet
        for comp, comp_data in install_cfg.items():
            cpack_comp_name = comp_data.get("name")
            source_dir = comp_data.get("source_dir")
            binary_dir = comp_data.get("binary_dir")
            install_dir = comp_data.get("install_dir")
            prefix = comp_data.get("prefix")

            # perform installation steps if we are not given a full path
            if not os.path.isabs(install_dir):
                # install_dir is just a sub-dir after install prefix, let's make a full path
                comp_data["install_dir"] = os.path.join(prefix, install_dir)

                # even perform a build in case of binary directory does not exist
                binary_dir = binary_dir if os.path.isabs(binary_dir) else os.path.join(self.build_temp, binary_dir)
                if not os.path.exists(binary_dir):
                    binary_dir = os.path.join(self.build_temp, binary_dir)
                    self.announce(f"Configuring {comp} cmake project", level=3)
                    self.spawn(["cmake", f"-DOpenVINODeveloperPackage_DIR={OPENVINO_BINARY_DIR}",
                                         f"-DPython3_EXECUTABLE={sys.executable}",
                                         f"-DCPACK_GENERATOR={CPACK_GENERATOR}",
                                         f"-DCMAKE_BUILD_TYPE={CONFIG}",
                                         "-DENABLE_WHEEL=OFF",
                                         self.cmake_args,
                                         "-S", source_dir,
                                         "-B", binary_dir])

                    self.announce(f"Building {comp} project", level=3)
                    self.spawn(["cmake", "--build", binary_dir,
                                         "--config", CONFIG,
                                         "--parallel", str(self.jobs)])

                self.announce(f"Installing {comp}", level=3)
                self.spawn(["cmake", "--install", binary_dir,
                                     "--prefix", prefix,
                                     "--config", CONFIG,
                                     "--component", cpack_comp_name])

    def run(self):
        # build and install clib into temporary directories
        if not PYTHON_EXTENSIONS_ONLY:
            self.cmake_build_and_install(LIB_INSTALL_CFG)

        # install python code into a temporary directory (site-packages)
        self.cmake_build_and_install(PY_INSTALL_CFG)

        # install clibs into a temporary directory (site-packages)
        if not PYTHON_EXTENSIONS_ONLY:
            self.run_command("build_clib")

        # Copy extra package_data content filtered by 'find_packages'
        exclude = ignore_patterns("*ez_setup*", "*__pycache__*", "*.egg-info*")
        src, dst = Path(PACKAGE_DIR), Path(self.build_lib)
        for path in src.glob("**/*"):
            if path.is_dir() or exclude(str(path)):
                continue
            path_rel = path.relative_to(src)
            (dst / path_rel.parent).mkdir(exist_ok=True, parents=True)
            copyfile(path, dst / path_rel)


class PrepareLibs(build_clib):
    """Prepares prebuilt libraries.

    The steps include: resolve symlinks for versioned shared libraries, set RPATHs for clibs
    """

    def run(self):
        """Run build_clib command."""
        # remove symlink to avoid copying it, set RPATH
        self.post_install(LIB_INSTALL_CFG)
        # copy clib to package data (to WHEEL_LIBS_INSTALL_DIR)
        self.copy_package_data(get_install_dirs_list(LIB_INSTALL_CFG))

    def post_install(self, install_cfg):
        """Install prebuilt libraries to the temp directories, set rpath."""
        for comp, comp_data in install_cfg.items():
            install_dir = comp_data.get("install_dir")

            # we need to resolve symlinks before setting rpath to avoid doing it multiple times
            install_dir_path = Path(install_dir)
            self.resolve_symlinks(install_dir_path)

            # set rpath if applicable
            if sys.platform != "win32" and comp_data.get("rpath"):
                # after tbb libraries on mac arm64 are signed, setting rpath for them will report error:
                # LC_SEGMENT_64 command 3 fileoff field plus filesize field extends past the end of the file
                if comp == "tbb_libs" and ARCH == "arm64" and sys.platform == "darwin":
                    continue

                for path in filter(
                    lambda x: any(item in ([".so"] if sys.platform == "linux" else [".dylib", ".so"])
                                  for item in x.suffixes), install_dir_path.glob("*"),
                ):
                    set_rpath(comp_data["rpath"], os.path.realpath(path))

    def get_reallink(self, link_file):
        """Get the real file path of a soft link."""
        real_name = link_file
        while True:
            real_name = os.readlink(real_name)
            if not os.path.isabs(real_name):
                real_name = os.path.join(os.path.dirname(link_file), real_name)
            if not Path(real_name).is_symlink():
                break
        return real_name

    def resolve_symlinks(self, local_base_dir: Path):
        """Resolves symlinks after installation via cmake install.

        Wheel package content must not contain symlinks. The block handles two kinds of soft links,
        take the library on Linux as an example.
        1. The first case: there are two soft links pointing to the real file,
        - input is libX.so->libX.so.Y and libX.so.Y->libX.so.Y.Z (e.g. hwloc library in oneTBB package).
        - input is libX.so->libX.so.Y.Z and libX.so.Y->libX.so.Y.Z (e.g. oneTBB library).
        2. The second case: there is one soft link pointing to the real file.
        Process results of the above two cases: remove soft links(libX.so and libX.so.Y), rename libX.so.Y.Z to libX.so.Y.
        """
        # step 1:
        # record real files and its symlinks {real file: soft link}
        # if there are two soft links pointing to the same file, like libX.so and libX.so.Y(including the above two cases),
        # only record the libX.so.Y and remove libX.so
        file_dict = {}
        for symlink in local_base_dir.rglob("*"):
            if symlink.is_symlink():
                real_name = self.get_reallink(symlink)
                if real_name in file_dict:
                    link_file_name_old = os.path.basename(file_dict[real_name])
                    link_file_name_new = os.path.basename(symlink)
                    if len(link_file_name_new) > len(link_file_name_old):
                        # replace libX.so/libX.dylib with libX.so.Y/libX.Y.dylib
                        self.announce(f"Unlink symlink {file_dict[real_name]}, use {symlink} instead", level=3)
                        os.unlink(file_dict[real_name])
                        file_dict[real_name] = symlink
                    else:
                        self.announce(f"Unlink symlink {symlink}, use {file_dict[real_name]} instead", level=3)
                        os.unlink(symlink)
                else:
                    file_dict[real_name] = symlink

        # step 2:
        # according to the corresponding relationship (file_dict),
        # remove the reserved soft link and rename the real file to the name of its soft link
        for real_name, symlink in file_dict.items():
            os.unlink(symlink)
            os.rename(real_name, symlink)
            self.announce(f"Resolved symlink {symlink} as {real_name}", level=3)

    def copy_package_data(self, src_dirs):
        """Collect package data files (clibs and other plugin support files) from preinstalled dirs and put all runtime libraries to the subpackage."""
        package_clibs_dir = os.path.join(PACKAGE_DIR, WHEEL_LIBS_INSTALL_DIR)
        os.makedirs(package_clibs_dir, exist_ok=True)

        for src_dir in src_dirs:
            # additional blacklist filter, just to fix cmake install issues
            blacklist_patterns = [  # static libraries and PBD files
                                    "^.*\\.a$", "^.*\\.lib$", "^.*\\.pdb$",
                                    # TBB debug libraries
                                    "^.*_debug\\.dll$", "^.*_debug\\.\\d*\\.dylib$", "^.*_debug\\.so\\.\\d*$",
                                    # hwloc static libs on Windows
                                    "^.*\\.la$"]

            # copy so / dylib files to WHEEL_LIBS_INSTALL_DIR (clibs) inside python package
            for file_path in Path(src_dir).rglob("*"):
                file_name = os.path.basename(file_path)
                if file_path.is_symlink():
                    # sanity check for self.resolve_symlinks
                    sys.exit(f"Wheel package content must not contain symlinks {file_path}")
                blacklisted = False
                for pattern in blacklist_patterns:
                    if re.match(pattern, file_name) is not None:
                        blacklisted = True
                        break
                if file_path.is_file() and not blacklisted:
                    dst_file = os.path.join(package_clibs_dir, file_name)
                    copyfile(file_path, dst_file)
                    self.announce(f"Copy {file_path} to {dst_file}", level=3)

        if Path(package_clibs_dir).exists():
            self.announce(f"Adding {WHEEL_LIBS_PACKAGE} package", level=3)
            packages.append(WHEEL_LIBS_PACKAGE)
            package_data.update({WHEEL_LIBS_PACKAGE: ["*"]})


def copy_file(src, dst, verbose=False, dry_run=False):
    """Custom file copy."""
    if dry_run:
        log.info(f"DRY RUN: Would copy '{src}' to '{dst}'")
    else:
        shutil.copyfile(src, dst)
        if verbose:
            log.info(f"Copied '{src}' to '{dst}'")


class CopyExt(build_ext):
    """Copy extension files to the build directory."""
    def run(self):
        if len(self.extensions) == 1:
            self.extensions = find_prebuilt_extensions(get_install_dirs_list(PY_INSTALL_CFG))

        for extension in self.extensions:
            if not isinstance(extension, PrebuiltExtension):
                raise SetupError(f"build_ext can accept PrebuiltExtension only, but got {extension.name}")
            src = extension.sources[0]
            dst = self.get_ext_fullpath(extension.name)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            # setting relative RPATH to found dlls
            if sys.platform != "win32" and not SKIP_RPATH:
                rpath = os.path.relpath(get_package_dir(PY_INSTALL_CFG), os.path.dirname(src))
                rpath = os.path.join(LIBS_RPATH, rpath, WHEEL_LIBS_INSTALL_DIR)
                set_rpath(rpath, os.path.realpath(src))

            copy_file(src, dst, verbose=self.verbose, dry_run=self.dry_run)


class CustomInstall(install):
    """Enable build_clib during the installation."""

    def run(self):
        self.run_command("build")
        install.run(self)


class CustomClean(Command):
    """Clean up staging directories."""

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def clean_install_prefix(self, install_cfg):
        for comp, comp_data in install_cfg.items():
            install_prefix = comp_data.get("prefix")
            self.announce(f"Cleaning {comp}: {install_prefix}", level=3)
            if os.path.exists(install_prefix):
                rmtree(install_prefix)

    def run(self):
        self.clean_install_prefix(LIB_INSTALL_CFG)
        self.clean_install_prefix(PY_INSTALL_CFG)


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
    """Remove rpath from macOS binaries.

    :param file_path: binary path
    :type file_path: pathlib.Path
    """
    if not sys.platform == "darwin":
        sys.exit(f"Unsupported platform: {sys.platform}")

    cmd = (
        f"otool -l {file_path} "  # noqa: P103
        f"| grep LC_RPATH -A3 "
        f"| grep -o 'path.*' "
        f"| cut -d ' ' -f2 "
        f"| xargs -I{{}} install_name_tool -delete_rpath {{}} {file_path}"
    )
    if os.WEXITSTATUS(os.system(cmd)) != 0:  # nosec
        sys.exit(f"Could not remove rpath for {file_path}")


def set_rpath(rpath, binary):
    """Setting rpath for Linux and macOS binaries."""
    print(f"Setting rpath {rpath} for {binary}")  # noqa: T001, T201
    cmd = []
    rpath_tool = ""

    if sys.platform == "linux":
        with open(os.path.realpath(binary), "rb") as file:
            if file.read(1) != b"\x7f":
                log.warning(f"WARNING: {binary}: missed ELF header")
                return
        rpath_tool = "patchelf"
        cmd = [rpath_tool, "--set-rpath", rpath, binary, "--force-rpath"]
    elif sys.platform == "darwin":
        rpath_tool = "install_name_tool"
        cmd = [rpath_tool, "-add_rpath", rpath, binary]
    else:
        sys.exit(f"Unsupported platform: {sys.platform}")

    if not is_tool(rpath_tool):
        sys.exit(f"Could not find {rpath_tool} on the system, " f"please make sure that this tool is installed")

    if sys.platform == "darwin":
        remove_rpath(binary)
    ret_info = subprocess.run(cmd, check=True, shell=False)  # nosec
    if ret_info.returncode != 0:
        sys.exit(f"Could not set rpath: {rpath} for {binary}")


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
    for search_dir in search_dirs:
        for path in Path(search_dir).glob(ext_pattern):
            # ignores usual inference plugins and libraries (clibs)
            if path.match("openvino/libs/*") or path.match(f"openvino/libs/openvino-{OPENVINO_VERSION}/*"):
                continue
            relpath = path.relative_to(search_dir)
            if relpath.parent != ".":
                package_names = str(relpath.parent).split(os.path.sep)
            else:
                package_names = []
            package_names.append(path.name.split(".", 1)[0])
            name = ".".join(package_names)
            extensions.append(PrebuiltExtension(name, sources=[str(path)]))
    if not extensions:
        # dummy extension to avoid python independent wheel
        extensions.append(PrebuiltExtension("openvino", sources=[str("setup.py")]))
    return extensions


def get_description(desc_file_path):
    """Read description from README.md."""
    with open(desc_file_path, "r", encoding="utf-8") as fstream:
        description = fstream.read()
    return description


def get_install_requires(requirements_file_path):
    """Read dependencies from requirements.txt."""
    with open(requirements_file_path, "r", encoding="utf-8") as fstream:
        install_requirements = fstream.read()
    return install_requirements


def get_install_dirs_list(install_cfg):
    """Collect all available directories with clibs or python extensions."""
    install_dirs = []
    for comp_info in install_cfg.values():
        full_install_dir = os.path.join(comp_info.get("prefix"), comp_info.get("install_dir"))
        if full_install_dir not in install_dirs:
            install_dirs.append(full_install_dir)
    return install_dirs


def get_package_dir(install_cfg):
    """Get python package path based on config. All the packages should be located in one directory."""
    py_package_path = ""
    dirs = get_install_dirs_list(install_cfg)
    if len(dirs) != 0:
        # setup.py support only one package directory, all modules should be located there
        py_package_path = dirs[0]
    return py_package_path


def find_entry_points(install_cfg):
    """Creates a list of entry points for OpenVINO runtime package."""
    entry_points = {
        "console_scripts": [],
    }
    for comp_info in install_cfg.values():
        empty_point = comp_info.get("entry_point")
        if empty_point is not None:
            entry_points["console_scripts"].append(empty_point["console_scripts"])
    return entry_points


def concat_files(input_files, output_file):
    """Concatenates multuple input files to a single output file."""
    with open(output_file, "w", encoding="utf-8") as outfile:
        for filename in input_files:
            with open(filename, "r", encoding="utf-8") as infile:
                content = infile.read()
                outfile.write(content)
    return output_file


OPENVINO_VERSION = WHEEL_VERSION = os.getenv("WHEEL_VERSION", "0.0.0")
PACKAGE_DIR = get_package_dir(PY_INSTALL_CFG)
# need to create package dir, because since https://github.com/pypa/wheel/commit/e43f2fcb296c2ac63e8bac2549ab596ab79accd0
# egg_info command works in this folder, because it's being created automatically
os.makedirs(PACKAGE_DIR, exist_ok=True)

packages = find_namespace_packages(PACKAGE_DIR)
package_data: typing.Dict[str, list] = {}
ext_modules = find_prebuilt_extensions(get_install_dirs_list(PY_INSTALL_CFG))
entry_points = find_entry_points(PY_INSTALL_CFG)

long_description_md = OPENVINO_SOURCE_DIR / "docs" / "dev" / "pypi_publish" / "pypi-openvino-rt.md"
md_files = [long_description_md, OPENVINO_SOURCE_DIR / "docs" / "dev" / "pypi_publish" / "pre-release-note.md"]
docs_url = "https://docs.openvino.ai/2023.0/index.html"

if os.getenv("CI_BUILD_DEV_TAG"):
    long_description_md = WORKING_DIR / "build" / "pypi-openvino-rt.md"
    long_description_md.parent.mkdir(exist_ok=True)
    concat_files(md_files, long_description_md)
    docs_url = "https://docs.openvino.ai/nightly/index.html"
    OPENVINO_VERSION = WHEEL_VERSION[0:8]

setup(
    name="openvino",
    version=WHEEL_VERSION,
    build=os.getenv("WHEEL_BUILD", "000"),
    author_email="openvino@intel.com",
    license="OSI Approved :: Apache Software License",
    author="Intel(R) Corporation",
    description="OpenVINO(TM) Runtime",
    install_requires=get_install_requires(SCRIPT_DIR.parents[0] / "requirements.txt"),
    long_description=get_description(long_description_md),
    long_description_content_type="text/markdown",
    download_url="https://github.com/openvinotoolkit/openvino/releases",
    url=docs_url,
    entry_points=entry_points,
    cmdclass={
        "build": CustomBuild,
        "install": CustomInstall,
        "build_clib": PrepareLibs,
        "build_ext": CopyExt,
        "clean": CustomClean,
    },
    ext_modules=ext_modules,
    packages=packages,
    package_dir={"": PACKAGE_DIR},
    package_data=package_data,
    zip_safe=False,
)
