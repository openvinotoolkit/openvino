# ******************************************************************************
# Copyright 2017-2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

import os
import pathlib
import shutil
import glob
import sysconfig
from sys import platform 

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.install_lib import install_lib


__version__ = os.environ.get("NGRAPH_VERSION", "0.0.0.dev0")
PYNGRAPH_ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
NGRAPH_ROOT_DIR = os.path.normpath(os.path.join(PYNGRAPH_ROOT_DIR, ".."))
OPENVINO_ROOT_DIR = os.path.normpath(os.path.join(PYNGRAPH_ROOT_DIR, "../.."))
# Change current working dircectory to ngraph/python
os.chdir(PYNGRAPH_ROOT_DIR)

NGRAPH_LIBS = ["ngraph", "onnx_importer"]

packages = [
    "ngraph",
    "ngraph.opset1",
    "ngraph.opset2",
    "ngraph.opset3",
    "ngraph.opset4",
    "ngraph.opset5",
    "ngraph.utils",
    "ngraph.impl",
    "ngraph.impl.op",
    "ngraph.impl.op.util",
    "ngraph.impl.passes",
]

data_files = []

with open(os.path.join(PYNGRAPH_ROOT_DIR, "requirements.txt")) as req:
    requirements = req.read().splitlines()


class CMakeExtension(Extension):

    def __init__(self, name, sources=[]):
        super().__init__(name=name, sources=sources)


class BuildCMakeExt(build_ext):

    user_options = build_ext.user_options + [
        ("config=", "c", "Build configuration [Release| Debug]")
    ]
    
    def initialize_options(self):
        self.config = None
        super().initialize_options()
    
    def finalize_options(self):
        if self.config not in ["Release", "Debug"]:
            self.config = "Release"
        super().finalize_options()
        
    def run(self):
        for extension in self.extensions:
            if extension.name == "_pyngraph":
                self.build_cmake(extension)

    def build_cmake(self, extension: Extension):
        """
        Steps required to build the extension
        """

        self.announce("Preparing the build environment", level=3)

        build_dir = pathlib.Path(self.build_temp)

        extension_path = pathlib.Path(self.get_ext_fullpath(extension.name))

        os.makedirs(build_dir, exist_ok=True)
        os.makedirs(extension_path.parent.absolute(), exist_ok=True)

        self.announce("Configuring cmake project", level=3)

        self.spawn(["cmake", "-H" + OPENVINO_ROOT_DIR, "-B" + self.build_temp,
                    "-DCMAKE_BUILD_TYPE=Release",
                    "-DENABLE_CLDNN=OFF",
                    "-DENABLE_OPENCV=OFF",
                    "-DENABLE_VPU=OFF",
                    "-DNGRAPH_PYTHON_BUILD_ENABLE=ON",
                    "-DNGRAPH_ONNX_IMPORT_ENABLE=ON"])
        
        self.announce("Building binaries", level=3)
        
        self.spawn(["cmake", "--build", self.build_temp, "--target", extension.name,
                    "--config", self.config])
       
        self.announce("Moving built python module to " + str(extension_path), level=3)
        bin_dir = os.path.join(OPENVINO_ROOT_DIR, 'bin')
        self.distribution.bin_dir = bin_dir

        pyds = [name for name in glob.iglob("{0}/**/{1}*{2}".format(bin_dir,
                extension.name,
                sysconfig.get_config_var("EXT_SUFFIX")), recursive=True)]
        for name in pyds:
            self.announce("copy " + os.path.join(bin_dir, name), level=3)
            shutil.copy(os.path.join(bin_dir, name),  extension_path)


class InstallCMakeLibs(install_lib):

    def run(self):
        """
        Copy libraries from the bin directory and place them as appropriate
        """

        self.announce("Adding library files", level=3)

        bin_dir = os.path.join(OPENVINO_ROOT_DIR, 'bin')

        lib_ext = ".so"
        if "linux" in platform or platform == "darwin":
            lib_ext = ".so"
        elif platform == "win32":
            lib_ext = ".dll"

        libs = []
        for ngraph_lib in NGRAPH_LIBS:
            libs.extend([name for name in
                         glob.iglob('{0}/**/*{1}{2}'.format(bin_dir, ngraph_lib, lib_ext), recursive=True)])
        self.distribution.data_files.extend([("lib", [os.path.normpath(os.path.join(bin_dir, lib)) for lib in libs])])
        self.distribution.run_command("install_data")
        super().run()


setup(
    name="ngraph-core",
    description="nGraph - Intel's graph compiler and runtime for Neural Networks",
    version=__version__,
    author="Intel Corporation",
    url="https://github.com/openvinotoolkit/openvino",
    license="License :: OSI Approved :: Apache Software License",
    ext_modules=[CMakeExtension(name="_pyngraph")],
    package_dir={"": "src"},
    packages=packages,
    install_requires=requirements,
    data_files=data_files,
    zip_safe=False,
    extras_require={},
    cmdclass={"build_ext": BuildCMakeExt,
              "install_lib": InstallCMakeLibs}
)

