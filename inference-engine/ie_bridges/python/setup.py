import subprocess
import os
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

PACKAGE = PACKAGE_NAME
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
        global  INFERENCE_ENGINE_DIR
        global  BUNDLE_INFERENCE_ENGINE

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
        pass

    def build_cmake_lib(self):
        pass

    @staticmethod
    def copy_compiled_libs(source_dir, destination):
        extensions = ['so', 'dll', 'pyd']
        for path in chain.from_iterable(source_dir.glob("*.%s" % ext) for ext in extensions):
            shutil.copy(path.as_posix(), destination.as_posix())


class clean(_clean):
    def run(self):
        super().run()


def paths_to_str(paths):
    return [p.as_posix() for p in paths]


with open(REQUIREMENTS_FILE) as reqs:
    requirements = set(reqs.read().splitlines())


c_sources = [
]

extensions = [
]

cmdclass = {
    'build_ext': build_ext,
    'build_py': build_py,
    'clean': clean,
    'install': install,
}

setup(
    name="inference_engine",
    version='0.1.1',
    description='Python inference for Inference Engine',
    packages=find_packages(exclude=['tests']),
    package_data={PACKAGE_NAME: ['*.so', '*.dll', '*dylib*', '*.pyd']},
    include_package_data=True,
    ext_modules=extensions,
    cmdclass=cmdclass,
    author='', author_email='',
    tests_require=['pytest'],
    install_requires=list(requirements),
    zip_safe=False,
)
