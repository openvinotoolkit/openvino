#!/usr/bin/env python3

# Copyright (C) 2018-2022 Intel Corporation SPDX-License-Identifier: Apache-2.0

""" Use this script to create a openvino-dev wheel package:
    $ python3 setup.py bdist_wheel
"""
# pylint: disable-msg=line-too-long

import os
import sys
import platform
import subprocess  # nosec
import shutil
from distutils import log
from distutils.command.build import build
from distutils.command.clean import clean
from pathlib import Path
from fnmatch import fnmatchcase
import pkg_resources
from setuptools.command.install import install
from setuptools import setup, find_namespace_packages

PYTHON_VERSION = f'python{sys.version_info.major}.{sys.version_info.minor}'
SCRIPT_DIR = Path(__file__).resolve().parents[0]
OPENVINO_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = SCRIPT_DIR / 'src'

PKG_INSTALL_CFG = {
    'openvino-mo': {
        'src_dir': OPENVINO_DIR / 'tools' / 'mo',
        'black_list': ['*unit_tests*'],
        'prefix': 'mo',
        'extract_entry_points': True,
        'extract_requirements': True,
        'extract_extras': True,
    },
    'benchmark_tool': {
        'src_dir': OPENVINO_DIR / 'tools' / 'benchmark_tool',
        'black_list': [],
        'prefix': 'benchmark_tool',
        'extract_entry_points': True,
        'extract_requirements': True,
    },
    "accuracy_checker": {
        'src_dir': OPENVINO_DIR / 'thirdparty' / 'open_model_zoo' / 'tools' / 'accuracy_checker',  # noqa:E501
        'black_list': ['*tests*'],
        'prefix': 'accuracy_checker',
        'extract_entry_points': True,
        'extract_requirements': True,
    },
    "omz_tools": {
        'src_dir': OPENVINO_DIR / 'thirdparty' / 'open_model_zoo' / 'tools' / 'model_tools',  # noqa:E501
        'black_list': [],
        'prefix': 'omz_tools',
        'extract_requirements': True,
        'extract_entry_points': True,
        'extract_extras': True,
    },
    "pot": {
        'src_dir': OPENVINO_DIR / 'tools' / 'pot',
        'black_list': ['*tests*'],
        'prefix': 'pot',
        'extract_entry_points': True,
        'extract_requirements': True,
    },
}


def ignore_patterns(*patterns):
    """
    Filter names by given patterns
    """
    return lambda name: any(fnmatchcase(name, pat=pat) for pat in patterns)


class CustomBuild(build):
    """Custom implementation of build"""

    def run(self):

        # pylint: disable-msg=too-many-locals
        self.announce('Installing packages', level=log.INFO)
        for cmp, cmp_data in PKG_INSTALL_CFG.items():
            self.announce(f'Processing package: {cmp}', level=log.INFO)
            if not cmp_data['src_dir'].is_dir():
                raise FileNotFoundError(
                    f'The source directory was not found: {cmp_data["src_dir"]}'
                )
            subprocess.call([sys.executable, 'setup.py', 'install',
                            '--root', str(SCRIPT_DIR),
                             '--prefix', str(cmp_data.get("prefix"))],
                            cwd=str(cmp_data.get('src_dir')))

            # grab installed modules
            lib_dir = 'lib/site-packages' if platform.system() == 'Windows' else f'lib/{PYTHON_VERSION}/site-packages'
            src = SCRIPT_DIR / cmp_data.get('prefix') / lib_dir

            egg_info = list(src.glob('**/*.egg-info'))
            if egg_info:

                def raw_req(req):
                    req.marker = None
                    return str(req)

                distributions = pkg_resources.find_distributions(str(Path(egg_info[0]).parent))
                for dist in distributions:
                    self.announce(f'Distribution: {dist.egg_name()}', level=log.INFO)

                    # load install_requires list
                    install_requires = list(sorted(map(raw_req, dist.requires())))
                    self.announce(f'Install requires: {install_requires}', level=log.INFO)
                    if cmp_data.get("extract_requirements"):
                        self.distribution.install_requires.extend(install_requires)

                    # load extras_require
                    if cmp_data.get("extract_extras"):
                        for extra in dist.extras:
                            if extra not in self.distribution.extras_require:
                                self.distribution.extras_require[extra] = []
                            extras_require = set(map(raw_req, dist.requires((extra,))))
                            self.announce(f'Extras: {extra}:{extras_require}', level=log.INFO)
                            self.distribution.extras_require[extra].extend(extras_require)

                    # extract console scripts
                    if cmp_data.get("extract_entry_points"):
                        for console_scripts in dist.get_entry_map('console_scripts'):
                            self.announce(f'Entry point: {console_scripts}', level=log.INFO)
                            entry = dist.get_entry_info('console_scripts', console_scripts)
                            self.distribution.entry_points['console_scripts'].append(str(entry))

            # copy modules to the build directory
            dst = Path(self.build_lib)
            black_list = cmp_data.get('black_list')
            exclude = ignore_patterns('*ez_setup*', '*__pycache__*', '*.egg-info*', *black_list)
            for path in src.glob('**/*'):
                if path.is_dir() or exclude(str(path)):
                    continue
                path_rel = path.relative_to(src)
                (dst / path_rel.parent).mkdir(exist_ok=True, parents=True)
                shutil.copyfile(path, dst / path_rel)

        # add dependecy on runtime package
        runtime_req = [f'openvino=={self.distribution.get_version()}']
        self.distribution.install_requires.extend(runtime_req)

        self.announce(f'{self.distribution.install_requires}', level=log.DEBUG)
        self.announce(f'{self.distribution.extras_require}', level=log.DEBUG)
        self.announce(f'{self.distribution.entry_points}', level=log.DEBUG)


class CustomInstall(install):
    """Enable build_clib during the installation"""

    def run(self):
        self.run_command('build')
        install.run(self)


class CustomClean(clean):
    """Clean up staging directories"""

    def clean(self, install_cfg):
        """Clean components staging directories"""
        for comp, comp_data in install_cfg.items():
            install_prefix = comp_data.get('prefix')
            self.announce(f'Cleaning {comp}: {install_prefix}', level=log.INFO)
            if os.path.exists(install_prefix):
                shutil.rmtree(install_prefix)

    def run(self):
        self.clean(PKG_INSTALL_CFG)
        for pattern in './build ./dist **/*.pyc **/*.tgz **/*.egg-info'.split(' '):
            paths = SCRIPT_DIR.glob(pattern)
            for path in paths:
                if path.is_file() and path.exists():
                    path = path.parent
                self.announce(f'Cleaning: {path}', level=log.INFO)
                shutil.rmtree(path)
        clean.run(self)


def get_description(desc_file_path):
    """read description from README.md"""
    with open(desc_file_path, 'r', encoding='utf-8') as fstream:
        description = fstream.read()
    return description

with (SCRIPT_DIR / 'requirements.txt').open() as requirements:
    install_reqs = [
        str(requirement)
        for requirement
        in pkg_resources.parse_requirements(requirements)
    ]


setup(
    name='openvino-dev',
    version=os.getenv('OPENVINO_VERSION', '0.0.0'),
    author='Intel® Corporation',
    license='OSI Approved :: Apache Software License',
    author_email='openvino_pushbot@intel.com',
    url='https://docs.openvinotoolkit.org/latest/index.html',
    download_url='https://github.com/openvinotoolkit/openvino/tags',
    description='OpenVINO(TM) Development Tools',
    long_description=get_description(SCRIPT_DIR.parents[1] / 'docs/install_guides/pypi-openvino-dev.md'),
    long_description_content_type='text/markdown',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    cmdclass={
        'build': CustomBuild,
        'install': CustomInstall,
        'clean': CustomClean,
    },
    entry_points = {
        'console_scripts': [],
    },
    install_requires=install_reqs,
    packages=find_namespace_packages(where=str(SRC_DIR)),
    package_dir={'': str(SRC_DIR)},
)
