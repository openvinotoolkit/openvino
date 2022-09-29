# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import pathlib
import shutil
import glob
import sysconfig
import sys
import multiprocessing

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.install_lib import install_lib
from setuptools.command.install import install as _install
from setuptools.command.develop import develop as _develop
from distutils.command.build import build as _build

__version__ = os.environ.get("NGRAPH_VERSION", "0.0.0.dev0")
PYTHON_API_ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
OPENVINO_ROOT_DIR = os.path.normpath(os.path.join(PYTHON_API_ROOT_DIR, "../../.."))
# Change current working directory to src/bindings/python
os.chdir(PYTHON_API_ROOT_DIR)

NGRAPH_LIBS = ["ngraph", "openvino"]

packages = [
    "ngraph",
    "ngraph.opset1",
    "ngraph.opset2",
    "ngraph.opset3",
    "ngraph.opset4",
    "ngraph.opset5",
    "ngraph.opset6",
    "ngraph.opset7",
    "ngraph.opset8",
    "ngraph.utils",
    "ngraph.impl",
    "ngraph.impl.op",
    "ngraph.impl.op.util",
    "ngraph.impl.passes",
    "openvino",
    "openvino.frontend",
    "openvino.preprocess",
    "openvino.offline_transformations",
    "openvino._offline_transformations",
    "openvino.runtime",
    "openvino.runtime.opset1",
    "openvino.runtime.opset2",
    "openvino.runtime.opset3",
    "openvino.runtime.opset4",
    "openvino.runtime.opset5",
    "openvino.runtime.opset6",
    "openvino.runtime.opset7",
    "openvino.runtime.opset8",
    "openvino.runtime.opset9",
    "openvino.runtime.utils",
    "openvino.runtime.op",
    "openvino.runtime.op.util",
    "openvino.runtime.passes",
]


data_files = []

with open(os.path.join(PYTHON_API_ROOT_DIR, "requirements.txt")) as req:
    requirements = req.read().splitlines()

cmdclass = {}
for super_class in [_build, _install, _develop]:
    class command(super_class):
        """Add user options for build, install and develop commands."""

        cmake_build_types = ["Release", "Debug", "RelWithDebInfo", "MinSizeRel"]
        user_options = super_class.user_options + [
            ("config=", None, f"Build configuration [{'|'.join(cmake_build_types)}]."),
            ("jobs=", None, "Specifies the number of jobs to use with make."),
            ("cmake-args=", None, "Additional options to be passed to CMake.")
        ]

        def initialize_options(self):
            """Set default values for all the options that this command supports."""
            super().initialize_options()
            self.config = None
            self.jobs = None
            self.cmake_args = None

    cmdclass[super_class.__name__] = command


class CMakeExtension(Extension):
    """Build extension stub."""

    def __init__(self, name, sources=None):
        if sources is None:
            sources = []
        super().__init__(name=name, sources=sources)


class BuildCMakeExt(build_ext):
    """Builds module using cmake instead of the python setuptools implicit build."""

    cmake_build_types = ["Release", "Debug", "RelWithDebInfo", "MinSizeRel"]
    user_options = [
        ("config=", None, f"Build configuration [{'|'.join(cmake_build_types)}]."),
        ("jobs=", None, "Specifies the number of jobs to use with make."),
        ("cmake-args=", None, "Additional options to be passed to CMake.")
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

        for cmd in ["build", "install", "develop"]:
            self.set_undefined_options(cmd, ("config", "config"),
                                       ("jobs", "jobs"),
                                       ("cmake_args", "cmake_args"))

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
                self.announce(f"Supported values: {', '.join(self.cmake_build_types)}", level=4)
                sys.exit(1)
        if self.jobs is None and os.getenv("MAX_JOBS") is not None:
            self.jobs = os.getenv("MAX_JOBS")
        self.jobs = multiprocessing.cpu_count() if self.jobs is None else int(self.jobs)

    def run(self):
        """Run CMake build for modules."""
        for extension in self.extensions:
            if extension.name == "pyopenvino":
                self.build_cmake(extension)
            if extension.name == "_pyngraph":
                self.build_cmake(extension)

    def build_cmake(self, extension: Extension):
        """Cmake configure and build steps."""
        self.announce("Preparing the build environment", level=3)
        plat_specifier = ".%s-%d.%d" % (self.plat_name, *sys.version_info[:2])
        self.build_temp = os.path.join(self.build_base, "temp" + plat_specifier, self.config)
        build_dir = pathlib.Path(self.build_temp)

        extension_path = pathlib.Path(self.get_ext_fullpath(extension.name))
        if extension.name == "pyopenvino":
            extension_path = pathlib.Path(os.path.join(extension_path.parent.absolute(), "openvino"))

        os.makedirs(build_dir, exist_ok=True)
        os.makedirs(extension_path.parent.absolute(), exist_ok=True)

        # If OpenVINO_DIR is set, try to build Python only,
        # otherwise build from scratch using OpenVINO root
        root_dir = OPENVINO_ROOT_DIR
        bin_dir = os.path.join(OPENVINO_ROOT_DIR, "bin")
        if os.environ.get("OpenVINO_DIR") is not None:
            root_dir = PYTHON_API_ROOT_DIR

        self.announce("Configuring cmake project", level=3)
        ext_args = self.cmake_args.split() if self.cmake_args else []
        ov_build_dir = os.path.join(OPENVINO_ROOT_DIR, "build")
        self.spawn(["cmake", "-S" + root_dir, "-B" + self.build_temp,
                    f"-DCMAKE_BUILD_TYPE={self.config}",
                    f"-DInferenceEngineDeveloperPackage_DIR={ov_build_dir}",
                    "-DENABLE_PYTHON=ON"] + ext_args)

        self.announce("Building binaries", level=3)

        self.spawn(["cmake", "--build", self.build_temp, "--target", extension.name,
                    "--config", self.config, "-j", str(self.jobs)])

        self.announce("Moving built python module to " + str(extension_path), level=3)
        pyds = list(glob.iglob(f"{bin_dir}/**/{extension.name}*{sysconfig.get_config_var('EXT_SUFFIX')}",
                               recursive=True))
        for name in pyds:
            self.announce("copy " + os.path.join(name), level=3)
            shutil.copy(name, extension_path)


class InstallCMakeLibs(install_lib):
    """Finds and installs NGraph libraries to a package location."""

    def run(self):
        """Copy libraries from the bin directory and place them as appropriate."""
        self.announce("Adding library files", level=3)

        root_dir = os.path.join(OPENVINO_ROOT_DIR, "bin")
        if os.environ.get("OpenVINO_DIR") is not None:
            root_dir = pathlib.Path(PYTHON_API_ROOT_DIR)

        lib_ext = ""
        if "linux" in sys.platform:
            lib_ext = ".so"
        elif sys.platform == "darwin":
            lib_ext = ".dylib"
        elif sys.platform == "win32":
            lib_ext = ".dll"

        libs = []
        print(root_dir)
        for ngraph_lib in NGRAPH_LIBS:
            libs.extend(list(glob.iglob(f"{root_dir}/**/*{ngraph_lib}*{lib_ext}", recursive=True)))
            print(libs)
        if not libs:
            raise Exception("NGraph libs not found.")

        self.announce("Adding library files" + str(libs), level=3)

        self.distribution.data_files.extend([("lib", [os.path.normpath(lib) for lib in libs])])
        self.distribution.run_command("install_data")
        super().run()


cmdclass["build_ext"] = BuildCMakeExt
cmdclass["install_lib"] = InstallCMakeLibs

setup(
    name="openvino",
    description="OpenVINO - deploying pre-trained deep learning models",
    version=__version__,
    author="Intel Corporation",
    url="https://github.com/openvinotoolkit/openvino",
    license="License :: OSI Approved :: Apache Software License",
    ext_modules=[CMakeExtension(name="_pyngraph"), CMakeExtension(name="pyopenvino")],
    package_dir={"": "src/compatibility", "openvino": "src/openvino"},
    packages=packages,
    install_requires=requirements,
    data_files=data_files,
    zip_safe=False,
    extras_require={},
    cmdclass=cmdclass
)
