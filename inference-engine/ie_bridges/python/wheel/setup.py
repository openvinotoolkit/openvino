# -*- coding: utf-8 -*-
# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os.path
import sys
import errno
import subprocess  # nosec
import typing
from pathlib import Path
from shutil import copyfile, rmtree
from distutils.command.install import install
from distutils.command.build import build
from distutils.command.clean import clean
from distutils.errors import DistutilsSetupError
from distutils.file_util import copy_file
from distutils import log
from setuptools import setup, find_namespace_packages, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.build_clib import build_clib
from decouple import config

WHEEL_LIBS_INSTALL_DIR = os.path.join('openvino', 'libs')
WHEEL_LIBS_PACKAGE = 'openvino.libs'
PYTHON_VERSION = f'python{sys.version_info.major}.{sys.version_info.minor}'

# The following variables can be defined in environment or .env file
CMAKE_BUILD_DIR = config('CMAKE_BUILD_DIR', '.')
CORE_LIBS_DIR = config('CORE_LIBS_DIR', '')
PLUGINS_LIBS_DIR = config('PLUGINS_LIBS_DIR', '')
NGRAPH_LIBS_DIR = config('NGRAPH_LIBS_DIR', '')
TBB_LIBS_DIR = config('TBB_LIBS_DIR', '')
PY_PACKAGES_DIR = config('PY_PACKAGES_DIR', '')
LIBS_RPATH = '$ORIGIN' if sys.platform == 'linux' else '@loader_path'

LIB_INSTALL_CFG = {
    'ie_libs': {
        'name': 'core',
        'prefix': 'libs.core',
        'install_dir': CORE_LIBS_DIR,
        'rpath': LIBS_RPATH,
    },
    'hetero_plugin': {
        'name': 'hetero',
        'prefix': 'libs.plugins',
        'install_dir': PLUGINS_LIBS_DIR,
        'rpath': LIBS_RPATH,
    },
    'gpu_plugin': {
        'name': 'gpu',
        'prefix': 'libs.plugins',
        'install_dir': PLUGINS_LIBS_DIR,
        'rpath': LIBS_RPATH,
    },
    'cpu_plugin': {
        'name': 'cpu',
        'prefix': 'libs.plugins',
        'install_dir': PLUGINS_LIBS_DIR,
        'rpath': LIBS_RPATH,
    },
    'multi_plugin': {
        'name': 'multi',
        'prefix': 'libs.plugins',
        'install_dir': PLUGINS_LIBS_DIR,
        'rpath': LIBS_RPATH,
    },
    'auto_plugin': {
        'name': 'auto',
        'prefix': 'libs.plugins',
        'install_dir': PLUGINS_LIBS_DIR,
        'rpath': LIBS_RPATH,
    },
    'myriad_plugin': {
        'name': 'myriad',
        'prefix': 'libs.plugins',
        'install_dir': PLUGINS_LIBS_DIR,
        'rpath': LIBS_RPATH,
    },
    'ngraph_libs': {
        'name': 'ngraph',
        'prefix': 'libs.ngraph',
        'install_dir': NGRAPH_LIBS_DIR,
        'rpath': LIBS_RPATH,
    },
    'tbb_libs': {
        'name': 'tbb',
        'prefix': 'libs.tbb',
        'install_dir': TBB_LIBS_DIR,
        'rpath': LIBS_RPATH,
    },
}

PY_INSTALL_CFG = {
    'ie_py': {
        'name': PYTHON_VERSION,
        'prefix': 'site-packages',
        'install_dir': PY_PACKAGES_DIR,
    },
    'ngraph_py': {
        'name': f'pyngraph_{PYTHON_VERSION}',
        'prefix': 'site-packages',
        'install_dir': PY_PACKAGES_DIR,
    },
}


class PrebuiltExtension(Extension):
    """Initialize Extension"""

    def __init__(self, name, sources, *args, **kwargs):
        if len(sources) != 1:
            nln = '\n'
            raise DistutilsSetupError(f'PrebuiltExtension can accept only one source, but got: {nln}{nln.join(sources)}')
        super().__init__(name, sources, *args, **kwargs)


class CustomBuild(build):
    """Custom implementation of build_clib"""

    def run(self):
        self.run_command('build_clib')
        build.run(self)


class CustomInstall(install):
    """Enable build_clib during the installation"""

    def run(self):
        self.run_command('build_clib')
        install.run(self)


class PrepareLibs(build_clib):
    """Prepare prebuilt libraries"""

    def run(self):
        self.configure(LIB_INSTALL_CFG)
        self.configure(PY_INSTALL_CFG)
        self.generate_package(get_dir_list(LIB_INSTALL_CFG))

    def configure(self, install_cfg):
        """Collect prebuilt libraries. Install them to the temp directories, set rpath."""
        for comp, comp_data in install_cfg.items():
            install_prefix = comp_data.get('prefix')
            install_dir = comp_data.get('install_dir')
            if install_dir and not os.path.isabs(install_dir):
                install_dir = os.path.join(install_prefix, install_dir)
                self.announce(f'Installing {comp}', level=3)
                self.spawn(['cmake', '--install', CMAKE_BUILD_DIR, '--prefix', install_prefix, '--component', comp_data.get('name')])
            # set rpath if applicable
            if sys.platform != 'win32' and comp_data.get('rpath'):
                file_types = ['.so'] if sys.platform == 'linux' else ['.dylib', '.so']
                for path in filter(lambda p: any(item in file_types for item in p.suffixes), Path(install_dir).glob('*')):
                    set_rpath(comp_data['rpath'], os.path.realpath(path))

    def generate_package(self, src_dirs):
        """
        Collect package data files from preinstalled dirs and
        put all runtime libraries to the subpackage
        """
        # additional blacklist filter, just to fix cmake install issues
        blacklist = ['.lib', '.pdb', '_debug.dll', '_debug.dylib']
        package_dir = os.path.join(get_package_dir(PY_INSTALL_CFG), WHEEL_LIBS_INSTALL_DIR)

        for src_dir in src_dirs:
            local_base_dir = Path(src_dir)
            for file_path in local_base_dir.rglob('*'):
                file_name = os.path.basename(file_path)
                if file_path.is_file() and not any(file_name.endswith(ext) for ext in blacklist):
                    dst_file = os.path.join(package_dir, os.path.relpath(file_path, local_base_dir))
                    os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                    copyfile(file_path, dst_file)

        if Path(package_dir).exists():
            self.announce(f'Adding {WHEEL_LIBS_PACKAGE} package', level=3)
            packages.append(WHEEL_LIBS_PACKAGE)
            package_data.update({WHEEL_LIBS_PACKAGE: ['*']})


class CopyExt(build_ext):
    """Copy extension files to the build directory"""

    def run(self):
        for extension in self.extensions:
            if not isinstance(extension, PrebuiltExtension):
                raise DistutilsSetupError(f'copy_ext can accept PrebuiltExtension only, but got {extension.name}')
            src = extension.sources[0]
            dst = self.get_ext_fullpath(extension.name)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            # setting relative path to find dlls
            if sys.platform != 'win32':
                rpath = os.path.relpath(get_package_dir(PY_INSTALL_CFG), os.path.dirname(src))
                if sys.platform == 'linux':
                    rpath = os.path.join('$ORIGIN', rpath, WHEEL_LIBS_INSTALL_DIR)
                elif sys.platform == 'darwin':
                    rpath = os.path.join('@loader_path', rpath, WHEEL_LIBS_INSTALL_DIR)
                set_rpath(rpath, os.path.realpath(src))

            copy_file(src, dst, verbose=self.verbose, dry_run=self.dry_run)


class CustomClean(clean):
    """Clean up staging directories"""

    def clean(self, install_cfg):
        for comp, comp_data in install_cfg.items():
            install_prefix = comp_data.get('prefix')
            self.announce(f'Cleaning {comp}: {install_prefix}', level=3)
            if os.path.exists(install_prefix):
                rmtree(install_prefix)

    def run(self):
        self.clean(LIB_INSTALL_CFG)
        self.clean(PY_INSTALL_CFG)
        clean.run(self)


def is_tool(name):
    """Check if the command-line tool is available"""
    try:
        devnull = subprocess.DEVNULL
        subprocess.Popen([name], stdout=devnull, stderr=devnull).communicate()  # nosec
    except OSError as error:
        if error.errno == errno.ENOENT:
            return False
    return True


def remove_rpath(file_path):
    """
    Remove rpath from binaries
    :param file_path: binary path
    :type file_path: pathlib.Path
    """
    if sys.platform == 'darwin':
        cmd = (
            f'otool -l {file_path} '  # noqa: P103
            f'| grep LC_RPATH -A3 '
            f'| grep -o "path.*" '
            f'| cut -d " " -f2 '
            f'| xargs -I{{}} install_name_tool -delete_rpath {{}} {file_path}'
        )
        if os.WEXITSTATUS(os.system(cmd)) != 0:  # nosec
            sys.exit(f'Could not remove rpath for {file_path}')
    else:
        sys.exit(f'Unsupported platform: {sys.platform}')


def set_rpath(rpath, executable):
    """Setting rpath for linux and macOS libraries"""
    print(f'Setting rpath {rpath} for {executable}')  # noqa: T001
    cmd = []
    rpath_tool = ''

    if sys.platform == 'linux':
        with open(os.path.realpath(executable), 'rb') as file:
            if file.read(1) != b'\x7f':
                log.warn(f'WARNING: {executable}: missed ELF header')
                return
        rpath_tool = 'patchelf'
        cmd = [rpath_tool, '--set-rpath', rpath, executable]
    elif sys.platform == 'darwin':
        rpath_tool = 'install_name_tool'
        cmd = [rpath_tool, '-add_rpath', rpath, executable]
    else:
        sys.exit(f'Unsupported platform: {sys.platform}')

    if is_tool(rpath_tool):
        if sys.platform == 'darwin':
            remove_rpath(executable)
        ret_info = subprocess.run(cmd, check=True, shell=False)  # nosec
        if ret_info.returncode != 0:
            sys.exit(f'Could not set rpath: {rpath} for {executable}')
    else:
        sys.exit(f'Could not found {rpath_tool} on the system, ' f'please make sure that this tool is installed')


def find_prebuilt_extensions(search_dirs):
    """collect prebuilt python extensions"""
    extensions = []
    ext_pattern = ''
    if sys.platform == 'linux':
        ext_pattern = '**/*.so'
    elif sys.platform == 'win32':
        ext_pattern = '**/*.pyd'
    elif sys.platform == 'darwin':
        ext_pattern = '**/*.so'
    for base_dir in search_dirs:
        for path in Path(base_dir).glob(ext_pattern):
            relpath = path.relative_to(base_dir)
            if relpath.parent != '.':
                package_names = str(relpath.parent).split(os.path.sep)
            else:
                package_names = []
            package_names.append(path.name.split('.', 1)[0])
            name = '.'.join(package_names)
            extensions.append(PrebuiltExtension(name, sources=[str(path)]))
    return extensions


def get_description(desc_file_path):
    """read description from README.md"""
    with open(desc_file_path, 'r', encoding='utf-8') as fstream:
        description = fstream.read()
    return description


def get_dependencies(requirements_file_path):
    """read dependencies from requirements.txt"""
    with open(requirements_file_path, 'r', encoding='utf-8') as fstream:
        dependencies = fstream.read()
    return dependencies


def get_dir_list(install_cfg):
    """collect all available directories with libs or python packages"""
    dirs = []
    for comp_info in install_cfg.values():
        cfg_prefix = comp_info.get('prefix')
        cfg_dir = comp_info.get('install_dir')
        if cfg_dir:
            if not os.path.isabs(cfg_dir):
                cfg_dir = os.path.join(cfg_prefix, cfg_dir)
            if cfg_dir not in dirs:
                dirs.append(cfg_dir)
    return dirs


def get_package_dir(install_cfg):
    """
    Get python package path based on config
    All the packages should be located in one directory
    """
    py_package_path = ''
    dirs = get_dir_list(install_cfg)
    if len(dirs) != 0:
        # setup.py support only one package directory, all modules should be located there
        py_package_path = dirs[0]
    return py_package_path


platforms = ['linux', 'win32', 'darwin']
if not any(pl in sys.platform for pl in platforms):
    sys.exit(f'Unsupported platform: {sys.platform}, expected: linux, win32, darwin')

# copy license file into the build directory
package_license = config('WHEEL_LICENSE', '')
if os.path.exists(package_license):
    copyfile(package_license, 'LICENSE')


packages = find_namespace_packages(','.join(get_dir_list(PY_INSTALL_CFG)))
package_data: typing.Dict[str, list] = {}

setup(
    version=config('WHEEL_VERSION', '0.0.0'),
    author_email=config('WHEEL_AUTHOR_EMAIL', 'openvino_pushbot@intel.com'),
    name=config('WHEEL_PACKAGE_NAME', 'openvino'),
    license=config('WHEEL_LICENCE_TYPE', 'OSI Approved :: Apache Software License'),
    author=config('WHEEL_AUTHOR', 'Intel Corporation'),
    description=config('WHEEL_DESC', 'Inference Engine Python* API'),
    install_requires=get_dependencies(config('WHEEL_REQUIREMENTS', 'requirements.txt')),
    long_description=get_description(config('WHEEL_OVERVIEW', 'pypi_overview.md')),
    long_description_content_type='text/markdown',
    download_url=config('WHEEL_DOWNLOAD_URL', 'https://github.com/openvinotoolkit/openvino/tags'),
    url=config('WHEEL_URL', 'https://docs.openvinotoolkit.org/latest/index.html'),
    cmdclass={
        'build': CustomBuild,
        'install': CustomInstall,
        'build_clib': PrepareLibs,
        'build_ext': CopyExt,
        'clean': CustomClean,
    },
    ext_modules=find_prebuilt_extensions(get_dir_list(PY_INSTALL_CFG)),
    packages=packages,
    package_dir={'': get_package_dir(PY_INSTALL_CFG)},
    package_data=package_data,
    zip_safe=False,
)
