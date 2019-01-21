import subprocess
from pathlib import Path
import platform
import sys
from itertools import chain
from distutils.command.build_py import build_py as _build_py
from distutils.command.clean import clean as _clean

import shutil
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.command.install import install as _install

IS_WINDOWS = (platform.system() == 'Windows')
IS_DARWIN = (platform.system() == 'Darwin')
IS_LINUX = (platform.system() == 'Linux')

REQUIREMENTS_FILE = 'requirements.txt'
PACKAGE_NAME = 'inference_engine'

PACKAGE = Path(PACKAGE_NAME)
C_LIB_NAME = '{}._C'.format(PACKAGE_NAME)

_build_cmd = ['cmake', '--build', '.']

INFERENCE_ENGINE_DIR = None
BUNDLE_INFERENCE_ENGINE = False


def parse_command_line_options(cls):
    """Propagates command line options to sub-commands.
    Allows to run install command with build_ext options"""

    base_user_options = getattr(cls, 'user_options', [])
    base_boolean_options = getattr(cls, 'boolean_options', [])
    base_run = cls.run
    base_init_options = cls.initialize_options

    cls.user_options = base_user_options + [
        ('copy-ie-libs', None, 'Copy Inference Engine Libraries to package directory'),
        ('inference-engine-dir=', None, 'Path to Inference Engine directory')
    ]

    cls.boolean_options = base_boolean_options + [
        'copy-ie-libs'
    ]

    def initialize_options(self):
        self.copy_ie_libs = False
        self.inference_engine_dir = None
        base_init_options(self)

    def run(self):
        global INFERENCE_ENGINE_DIR
        global BUNDLE_INFERENCE_ENGINE

        if self.copy_ie_libs:
            BUNDLE_INFERENCE_ENGINE = True

        if self.inference_engine_dir:
            INFERENCE_ENGINE_DIR = self.inference_engine_dir

        base_run(self)

    cls.initialize_options = initialize_options
    cls.run = run
    return cls


@parse_command_line_options
class install(_install):
    pass


@parse_command_line_options
class build_py(_build_py):
    pass


@parse_command_line_options
class build_ext(_build_ext):
    def run(self):
        if not self.extensions:
            return

        for i, ext in enumerate(self.extensions):
            if ext.name == C_LIB_NAME:
                self._build_cmake()
                self.extensions.pop(i)
                break

        super().run()

    def _build_cmake(self):
        print("Building C++ extension")
        if Path.cwd().joinpath("Makefile").is_file():
            # in build directory, run make only
            subprocess.call(_build_cmd)
        else:
            # compile extension library and
            self.build_cmake_lib()
        print("Built C++ extension")

    def build_cmake_lib(self):
        def save_call(*args, error_msg=None, **kwargs):
            if subprocess.call(*args, **kwargs) != 0:
                if error_msg:
                    print(error_msg)
                shutil.rmtree(tmp_build_dir.as_posix(), ignore_errors=True)
                sys.exit(1)

        tmp_build_dir = Path("tmp_build")
        destination = Path(self.build_lib) / PACKAGE_NAME if not self.inplace else Path(PACKAGE_NAME)
        tmp_build_dir.mkdir(exist_ok=False)

        _python_executable_opt = ['-DPYTHON_EXECUTABLE={}'.format(sys.executable)]
        _build_type_opt = ['-DCMAKE_BUILD_TYPE=Release']
        _generator_opt = ['-G', 'NMake Makefiles' if IS_WINDOWS else "Unix Makefiles"]

        _optional = []
        if BUNDLE_INFERENCE_ENGINE:
            _optional.append('-DCOPY_IE_LIBS=ON')

        if INFERENCE_ENGINE_DIR:
            _optional.append('-DInferenceEngine_DIR={}'.format(INFERENCE_ENGINE_DIR))

        _cmake_cmd = list(chain(['cmake'], _generator_opt, _build_type_opt, _python_executable_opt, _optional, ['..']))

        save_call(_cmake_cmd, cwd=tmp_build_dir.as_posix(), error_msg="Cmake generator failed")
        save_call(_build_cmd, cwd=tmp_build_dir.as_posix(), error_msg="Build command failed")

        build_ext.copy_compiled_libs(tmp_build_dir / PACKAGE_NAME, destination)
        shutil.rmtree(tmp_build_dir.as_posix(), ignore_errors=False)

    @staticmethod
    def copy_compiled_libs(source_dir, destination):
        extensions = ['so', 'dll', 'pyd']
        for path in chain.from_iterable(source_dir.glob("*.%s" % ext) for ext in extensions):
            shutil.copy(path.as_posix(), destination.as_posix())


class clean(_clean):
    def run(self):
        shutil.rmtree("tmp_build", ignore_errors=True)
        extensions = ['so', 'dll', 'pyd']
        for path in chain.from_iterable(PACKAGE.glob("*.%s" % ext) for ext in extensions):
            path.unlink()
        super().run()


def paths_to_str(paths):
    return [p.as_posix() for p in paths]


with open(REQUIREMENTS_FILE) as reqs:
    requirements = set(reqs.read().splitlines())

# do not spoil pre-installed opencv (in case it was built from source)
_opencv_package = "opencv-python"
try:
    import cv2

    if _opencv_package in requirements:
        requirements.remove(_opencv_package)
except ImportError:
    requirements.add(_opencv_package)


c_sources = [
    PACKAGE / 'ie_driver.cpp',
    PACKAGE / 'ie_driver.hpp',

    PACKAGE / 'c_ie_driver.pxd',
    PACKAGE / 'ie_driver.pyx',
    PACKAGE / 'ie_driver.pxd',
]

extensions = [
    Extension(C_LIB_NAME, paths_to_str(c_sources))
]

cmdclass = {
    'build_ext': build_ext,
    'build_py': build_py,
    'clean': clean,
    'install': install,
}

setup(
    name="src",
    version='1.0',
    description='Python inference for Inference Engine',
    packages=find_packages(exclude=['tests']),
    package_data={PACKAGE_NAME: ['*.so', '*.dll', '*dylib*', '*.pyd']},
    include_package_data=True,
    ext_modules=extensions,
    cmdclass=cmdclass,
    install_requires=list(requirements),
    zip_safe=False,
)
