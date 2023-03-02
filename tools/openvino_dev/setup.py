#!/usr/bin/env python3

# Copyright (C) 2018-2023 Intel Corporation SPDX-License-Identifier: Apache-2.0

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
        BUILD_BASE = Path.cwd() / self.build_base
        for cmp, cmp_data in PKG_INSTALL_CFG.items():
            self.announce(f'Processing package: {cmp}', level=log.INFO)
            if not cmp_data['src_dir'].is_dir():
                raise FileNotFoundError(
                    f'The source directory was not found: {cmp_data["src_dir"]}'
                )
            subprocess.call([sys.executable, 'setup.py', 'install',
                            '--root', str(BUILD_BASE),
                            '--prefix', str(cmp_data.get("prefix"))],
                            cwd=str(cmp_data.get('src_dir')))

            # grab installed modules
            lib_dir = 'lib/site-packages' if platform.system() == 'Windows' else f'lib/{PYTHON_VERSION}/site-packages'
            src = BUILD_BASE / cmp_data.get('prefix') / lib_dir

            egg_info = list(src.glob('**/*.egg-info'))
            if egg_info:
                distributions = pkg_resources.find_distributions(str(Path(egg_info[0]).parent))
                for dist in distributions:
                    self.announce(f'Distribution: {dist.egg_name()}', level=log.INFO)
                    dmap = dist._build_dep_map() # pylint: disable=W0212

                    # load install_requires list
                    if cmp_data.get("extract_requirements"):
                        # install requires {None: [requirements]}
                        install_requires = sorted(map(str, dmap.get(None, [])))
                        self.announce(f'Install requires: {install_requires}', level=log.INFO)
                        self.distribution.install_requires.extend(install_requires)
                        # conditional requirements {':<condition>': [requirements]}
                        conditionals_req = dict(filter(lambda x: x[0] is not None and x[0].split(':')[0] == '', dmap.items()))
                        self.announce(f'Install requires with marker: {conditionals_req}', level=log.INFO)
                        for extra, req in conditionals_req.items():
                            if extra not in self.distribution.extras_require:
                                self.distribution.extras_require[extra] = []
                            self.distribution.extras_require[extra].extend(sorted(map(str, req)))

                    if cmp_data.get("extract_extras"):
                        # extra requirements {'marker:<condition>': [requirements]}
                        extras = dict(filter(lambda x: x[0] is not None and x[0].split(':')[0] != '', dmap.items()))
                        for extra, req in extras.items():
                            self.announce(f'Extras: {extra}:{req}', level=log.INFO)
                            if extra not in self.distribution.extras_require:
                                self.distribution.extras_require[extra] = []
                            self.distribution.extras_require[extra].extend(sorted(map(str, req)))

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

        # remove duplications in requirements
        reqs_set = set(map(lambda x: x.lower(), self.distribution.install_requires))
        self.distribution.install_requires = sorted(reqs_set)
        for extra, req in self.distribution.extras_require.items():
            unique_req = list(set(map(lambda x: x.lower(), req)))
            self.distribution.extras_require[extra] = unique_req

        # add dependency on runtime package
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


def concat_files(output_file, input_files):
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for filename in input_files:
            with open(filename, 'r', encoding='utf-8') as infile:
                content = infile.read()
                outfile.write(content)
    return output_file


description_md = SCRIPT_DIR.parents[1] / 'docs' / 'install_guides' / 'pypi-openvino-dev.md'
md_files = [description_md, SCRIPT_DIR.parents[1] / 'docs' / 'install_guides' / 'pre-release-note.md']
docs_url = 'https://docs.openvino.ai/latest/index.html'

if(os.getenv('CI_BUILD_DEV_TAG')):
    output = Path.cwd() / 'build' / 'pypi-openvino-dev.md'
    output.parent.mkdir(exist_ok=True)
    description_md = concat_files(output, md_files)
    docs_url = 'https://docs.openvino.ai/nightly/index.html'

setup(
    name='openvino-dev',
    version=os.getenv('OPENVINO_VERSION', '0.0.0'),
    author=os.getenv('WHEEL_AUTHOR', 'Intel® Corporation'),
    license=os.getenv('WHEEL_LICENCE_TYPE', 'OSI Approved :: Apache Software License'),
    author_email=os.getenv('WHEEL_AUTHOR_EMAIL', 'openvino_pushbot@intel.com'),
    url=os.getenv('WHEEL_URL', docs_url),
    download_url=os.getenv('WHEEL_DOWNLOAD_URL', 'https://github.com/openvinotoolkit/openvino/tags'),
    description=os.getenv('WHEEL_DESC', 'OpenVINO(TM) Development Tools'),
    long_description=get_description(os.getenv('WHEEL_OVERVIEW', description_md)),
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
