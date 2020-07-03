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

import distutils.ccompiler
import os
import re
import sys

import setuptools
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

__version__ = os.environ.get("NGRAPH_VERSION", "0.0.0.dev0")
PYNGRAPH_ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
PYNGRAPH_SRC_DIR = os.path.join(PYNGRAPH_ROOT_DIR, "src")
NGRAPH_DEFAULT_INSTALL_DIR = os.environ.get("HOME")
NGRAPH_ONNX_IMPORT_ENABLE = os.environ.get("NGRAPH_ONNX_IMPORT_ENABLE")
NGRAPH_PYTHON_DEBUG = os.environ.get("NGRAPH_PYTHON_DEBUG")


def find_ngraph_dist_dir():
    """Return location of compiled ngraph library home."""
    if os.environ.get("NGRAPH_CPP_BUILD_PATH"):
        ngraph_dist_dir = os.environ.get("NGRAPH_CPP_BUILD_PATH")
    else:
        ngraph_dist_dir = os.path.join(NGRAPH_DEFAULT_INSTALL_DIR, "ngraph_dist")

    found = os.path.exists(os.path.join(ngraph_dist_dir, "include/ngraph"))
    if not found:
        print(
            "Cannot find nGraph library in {} make sure that "
            "NGRAPH_CPP_BUILD_PATH is set correctly".format(ngraph_dist_dir)
        )
        sys.exit(1)
    else:
        print("nGraph library found in {}".format(ngraph_dist_dir))
        return ngraph_dist_dir


def find_pybind_headers_dir():
    """Return location of pybind11 headers."""
    if os.environ.get("PYBIND_HEADERS_PATH"):
        pybind_headers_dir = os.environ.get("PYBIND_HEADERS_PATH")
    else:
        pybind_headers_dir = os.path.join(PYNGRAPH_ROOT_DIR, "pybind11")

    found = os.path.exists(os.path.join(pybind_headers_dir, "include/pybind11"))
    if not found:
        print(
            "Cannot find pybind11 library in {} make sure that "
            "PYBIND_HEADERS_PATH is set correctly".format(pybind_headers_dir)
        )
        sys.exit(1)
    else:
        print("pybind11 library found in {}".format(pybind_headers_dir))
        return pybind_headers_dir


NGRAPH_CPP_DIST_DIR = find_ngraph_dist_dir()
PYBIND11_INCLUDE_DIR = find_pybind_headers_dir() + "/include"
NGRAPH_CPP_INCLUDE_DIR = NGRAPH_CPP_DIST_DIR + "/include"
if os.path.exists(os.path.join(NGRAPH_CPP_DIST_DIR, "lib")):
    NGRAPH_CPP_LIBRARY_DIR = os.path.join(NGRAPH_CPP_DIST_DIR, "lib")
elif os.path.exists(os.path.join(NGRAPH_CPP_DIST_DIR, "lib64")):
    NGRAPH_CPP_LIBRARY_DIR = os.path.join(NGRAPH_CPP_DIST_DIR, "lib64")
else:
    print(
        "Cannot find library directory in {}, make sure that nGraph is installed "
        "correctly".format(NGRAPH_CPP_DIST_DIR)
    )
    sys.exit(1)

if sys.platform == "win32":
    NGRAPH_CPP_DIST_DIR = os.path.normpath(NGRAPH_CPP_DIST_DIR)
    PYBIND11_INCLUDE_DIR = os.path.normpath(PYBIND11_INCLUDE_DIR)
    NGRAPH_CPP_INCLUDE_DIR = os.path.normpath(NGRAPH_CPP_INCLUDE_DIR)
    NGRAPH_CPP_LIBRARY_DIR = os.path.normpath(NGRAPH_CPP_LIBRARY_DIR)

NGRAPH_CPP_LIBRARY_NAME = "ngraph"
"""For some platforms OpenVINO adds 'd' suffix to library names in debug configuration"""
if len([fn for fn in os.listdir(NGRAPH_CPP_LIBRARY_DIR) if re.search("ngraphd", fn)]):
    NGRAPH_CPP_LIBRARY_NAME = "ngraphd"

ONNX_IMPORTER_CPP_LIBRARY_NAME = "onnx_importer"
if len([fn for fn in os.listdir(NGRAPH_CPP_LIBRARY_DIR) if re.search("onnx_importerd", fn)]):
    ONNX_IMPORTER_CPP_LIBRARY_NAME = "onnx_importerd"


def parallelCCompile(
    self,
    sources,
    output_dir=None,
    macros=None,
    include_dirs=None,
    debug=0,
    extra_preargs=None,
    extra_postargs=None,
    depends=None,
):
    """Build sources in parallel.

    Reference link:
    http://stackoverflow.com/questions/11013851/speeding-up-build-process-with-distutils
    Monkey-patch for parallel compilation.
    """
    # those lines are copied from distutils.ccompiler.CCompiler directly
    macros, objects, extra_postargs, pp_opts, build = self._setup_compile(
        output_dir, macros, include_dirs, sources, depends, extra_postargs
    )
    cc_args = self._get_cc_args(pp_opts, debug, extra_preargs)

    if NGRAPH_PYTHON_DEBUG in ["TRUE", "ON", True]:
        try:
            # pybind11 is much more verbose without -DNDEBUG
            self.compiler.remove("-DNDEBUG")
            self.compiler.remove("-O2")
            self.compiler_so.remove("-DNDEBUG")
            self.compiler_so.remove("-O2")
        except (AttributeError, ValueError):
            pass
    # parallel code
    import multiprocessing.pool

    def _single_compile(obj):
        try:
            src, ext = build[obj]
        except KeyError:
            return
        self._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)

    # convert to list, imap is evaluated on-demand
    pool = multiprocessing.pool.ThreadPool()
    list(pool.imap(_single_compile, objects))
    return objects


distutils.ccompiler.CCompiler.compile = parallelCCompile


def has_flag(compiler, flagname):
    """Check whether a flag is supported by the specified compiler.

    As of Python 3.6, CCompiler has a `has_flag` method.
    cf http://bugs.python.org/issue26689
    """
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".cpp") as f:
        f.write("int main (int argc, char **argv) { return 0; }")
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def cpp_flag(compiler):
    """Check and return the -std=c++11 compiler flag."""
    if sys.platform == "win32":
        return ""  # C++11 is on by default in MSVC
    elif has_flag(compiler, "-std=c++11"):
        return "-std=c++11"
    else:
        raise RuntimeError("Unsupported compiler -- C++11 support is needed!")


sources = [
    "pyngraph/axis_set.cpp",
    "pyngraph/axis_vector.cpp",
    "pyngraph/coordinate.cpp",
    "pyngraph/coordinate_diff.cpp",
    "pyngraph/dict_attribute_visitor.cpp",
    "pyngraph/dimension.cpp",
    "pyngraph/function.cpp",
    "pyngraph/node.cpp",
    "pyngraph/node_factory.cpp",
    "pyngraph/ops/constant.cpp",
    "pyngraph/ops/get_output_element.cpp",
    "pyngraph/ops/op.cpp",
    "pyngraph/ops/parameter.cpp",
    "pyngraph/ops/regmodule_pyngraph_op.cpp",
    "pyngraph/ops/result.cpp",
    "pyngraph/ops/util/arithmetic_reduction.cpp",
    "pyngraph/ops/util/binary_elementwise_arithmetic.cpp",
    "pyngraph/ops/util/binary_elementwise_comparison.cpp",
    "pyngraph/ops/util/binary_elementwise_logical.cpp",
    "pyngraph/ops/util/index_reduction.cpp",
    "pyngraph/ops/util/op_annotations.cpp",
    "pyngraph/ops/util/regmodule_pyngraph_op_util.cpp",
    "pyngraph/ops/util/unary_elementwise_arithmetic.cpp",
    "pyngraph/passes/manager.cpp",
    "pyngraph/passes/regmodule_pyngraph_passes.cpp",
    "pyngraph/partial_shape.cpp",
    "pyngraph/pyngraph.cpp",
    "pyngraph/serializer.cpp",
    "pyngraph/shape.cpp",
    "pyngraph/strides.cpp",
    "pyngraph/tensor_iterator_builder.cpp",
    "pyngraph/types/element_type.cpp",
    "pyngraph/types/regmodule_pyngraph_types.cpp",
    "pyngraph/util.cpp",
]

packages = [
    "ngraph",
    "ngraph.utils",
    "ngraph.impl",
    "ngraph.impl.op",
    "ngraph.impl.op.util",
    "ngraph.impl.passes",
]

sources = [PYNGRAPH_SRC_DIR + "/" + source for source in sources]

include_dirs = [PYNGRAPH_SRC_DIR, NGRAPH_CPP_INCLUDE_DIR, PYBIND11_INCLUDE_DIR]

library_dirs = [NGRAPH_CPP_LIBRARY_DIR]

libraries = [NGRAPH_CPP_LIBRARY_NAME, ONNX_IMPORTER_CPP_LIBRARY_NAME]

extra_compile_args = []
if NGRAPH_ONNX_IMPORT_ENABLE in ["TRUE", "ON", True]:
    extra_compile_args.append("-DNGRAPH_ONNX_IMPORT_ENABLE")

extra_link_args = []

data_files = [
    (
        "lib",
        [
            os.path.join(NGRAPH_CPP_LIBRARY_DIR, library)
            for library in os.listdir(NGRAPH_CPP_LIBRARY_DIR)
        ],
    ),
    (
        "licenses",
        [
            os.path.join(NGRAPH_CPP_DIST_DIR, "licenses", license)
            for license in os.listdir(os.path.join(NGRAPH_CPP_DIST_DIR, "licenses"))
        ],
    ),
    ("", [os.path.join(NGRAPH_CPP_DIST_DIR, "LICENSE")],),
]

if NGRAPH_ONNX_IMPORT_ENABLE in ["TRUE", "ON", True]:
    onnx_sources = [
        "pyngraph/onnx_import/onnx_import.cpp",
    ]
    onnx_sources = [PYNGRAPH_SRC_DIR + "/" + source for source in onnx_sources]
    sources = sources + onnx_sources

    packages.append("ngraph.impl.onnx_import")

ext_modules = [
    Extension(
        "_pyngraph",
        sources=sources,
        include_dirs=include_dirs,
        define_macros=[("VERSION_INFO", __version__)],
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++",
    ),
]


def add_platform_specific_link_args(link_args):
    """Add linker flags specific for the OS detected during the build."""
    if sys.platform.startswith("linux"):
        link_args += ["-Wl,-rpath,$ORIGIN/../.."]
        link_args += ["-z", "noexecstack"]
        link_args += ["-z", "relro"]
        link_args += ["-z", "now"]
    elif sys.platform == "darwin":
        link_args += ["-Wl,-rpath,@loader_path/../.."]
        link_args += ["-stdlib=libc++"]
    elif sys.platform == "win32":
        link_args += ["/LTCG"]


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""

    def _add_extra_compile_arg(self, flag, compile_args):
        """Return True if successfully added given flag to compiler args."""
        if has_flag(self.compiler, flag):
            compile_args += [flag]
            return True
        return False

    def _add_debug_or_release_flags(self):
        """Return compiler flags for Release and Debug build types."""
        if NGRAPH_PYTHON_DEBUG in ["TRUE", "ON", True]:
            if sys.platform == "win32":
                return ["/Od", "/Zi", "/RTC1"]
            else:
                return ["-O0", "-g"]
        else:
            if sys.platform == "win32":
                return ["/O2"]
            else:
                return ["-O2", "-D_FORTIFY_SOURCE=2"]

    def _add_win_compiler_flags(self, ext):
        self._add_extra_compile_arg("/GL", ext.extra_compile_args)  # Whole Program Optimization
        self._add_extra_compile_arg("/analyze", ext.extra_compile_args)

    def _add_unix_compiler_flags(self, ext):
        if not self._add_extra_compile_arg("-fstack-protector-strong", ext.extra_compile_args):
            self._add_extra_compile_arg("-fstack-protector", ext.extra_compile_args)

        self._add_extra_compile_arg("-fvisibility=hidden", ext.extra_compile_args)
        self._add_extra_compile_arg("-flto", ext.extra_compile_args)
        self._add_extra_compile_arg("-fPIC", ext.extra_compile_args)

        ext.extra_compile_args += ["-Wformat", "-Wformat-security"]

    def _customize_compiler_flags(self):
        """Modify standard compiler flags."""
        try:
            # -Wstrict-prototypes is not a valid option for c++
            self.compiler.compiler_so.remove("-Wstrict-prototypes")
            if NGRAPH_PYTHON_DEBUG in ["TRUE", "ON", True]:
                # pybind11 is much more verbose without -DNDEBUG
                self.compiler.compiler_so.remove("-DNDEBUG")
                self.compiler.compiler_so.remove("-O2")
        except (AttributeError, ValueError):
            pass

    def build_extensions(self):
        """Build extension providing extra compiler flags."""
        self._customize_compiler_flags()
        for ext in self.extensions:
            ext.extra_compile_args += [cpp_flag(self.compiler)]

            if sys.platform == "win32":
                self._add_win_compiler_flags(ext)
            else:
                self._add_unix_compiler_flags(ext)

            add_platform_specific_link_args(ext.extra_link_args)

            ext.extra_compile_args += self._add_debug_or_release_flags()

            if sys.platform == "darwin":
                ext.extra_compile_args += ["-stdlib=libc++"]

        build_ext.build_extensions(self)


with open(os.path.join(PYNGRAPH_ROOT_DIR, "requirements.txt")) as req:
    requirements = req.read().splitlines()

setup(
    name="ngraph-core",
    description="nGraph - Intel's graph compiler and runtime for Neural Networks",
    version=__version__,
    author="Intel Corporation",
    url="https://github.com/openvinotoolkit/openvino",
    license="License :: OSI Approved :: Apache Software License",
    ext_modules=ext_modules,
    package_dir={'': 'src'},
    packages=packages,
    cmdclass={"build_ext": BuildExt},
    data_files=data_files,
    install_requires=requirements,
    zip_safe=False,
    extras_require={},
)
