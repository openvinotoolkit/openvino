#!/usr/bin/env python3

# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Use this script to create a wheel with Model Optimizer code:

$ python setup.py sdist bdist_wheel
"""

import os
import re
from pathlib import Path
from shutil import copyfile, copy

from setuptools import setup, find_namespace_packages
from setuptools.command.build_py import build_py
from setuptools.command.install import install

from typing import Dict, List

prefix = 'openvino/tools/mo/'
SETUP_DIR = Path(__file__).resolve().parent / Path(prefix)


def read_constraints(path: str='../constraints.txt') -> Dict[str, List[str]]:
    """
    Read a constraints.txt file and return a dict
    of {package_name: [required_version_1, required_version_2]}.
    The dict values are a list because a package can be mentioned
    multiple times, for example:
        mxnet~=1.2.0; sys_platform == 'win32'
        mxnet>=1.7.0; sys_platform != 'win32'
    """
    constraints = {}
    with open(Path(__file__).resolve().parent / path) as f:
        raw_constraints = f.readlines()
    for line in raw_constraints:
        # skip comments
        if line.startswith('#'):
            continue
        line = line.replace('\n', '')
        # read constraints for that package
        package, delimiter, constraint = re.split('(~|=|<|>|;)', line, maxsplit=1)
        # if there is no entry for that package, add it
        if constraints.get(package) is None:
            constraints[package] = [delimiter + constraint]
        # else add another entry for that package
        else:
            constraints[package].extend([delimiter + constraint])
    return constraints


def read_requirements(path: str) -> List[str]:
    """
    Read a requirements.txt file and return a list
    of requirements. Three cases are supported, the
    list corresponds to priority:
    1. version specified in requirements.txt
    2. version specified in constraints.txt
    3. version unbound

    Putting environment markers into constraints.txt is prone to bugs.
    They should be specified in requirements.txt files.
    """
    requirements = []
    constraints = read_constraints()
    with open(Path(__file__).resolve().parent / path) as f:
        raw_requirements = f.readlines()
    for line in raw_requirements:
        # skip comments and constraints link
        if line.startswith(('#', '-c')):
            continue
        # get rid of newlines
        line = line.replace('\n', '')
        # if version is specified (non-word chars present) 
        package_constraint = constraints.get(line.split(';')[0])
        if re.search('(~|=|<|>)', line) and len(line.split(';'))>1:
            if package_constraint:  # both markers and versions specified
                marker_index = line.find(";")
                # insert package version between package name and environment markers
                line = line[:marker_index] \
                + ",".join([constraint for constraint in package_constraint]) \
                + line[marker_index:]
            requirements.append(line)
        # else get version from constraints
        else:
            constraint = constraints.get(line)
            # if version found in constraints.txt
            if constraint:
                for marker in constraint:
                    requirements.append(line+marker)
            # else version is unbound
            else:
                requirements.append(line)
    return requirements


# Detect all the framework specific requirements_*.txt files.
requirements_txt = []
py_modules = []
for item in os.listdir():
    if re.match(r'requirements_?(tf|tf2|onnx|kaldi|caffe)?\.txt', item):
        requirements_txt.append(item)
for item in os.listdir(prefix):
    if re.match(r'mo(.*)\.py|main(.*)\.py', item):
        py_modules.append(prefix.replace('/', '.') + item.split('.')[0])
py_modules.append(prefix.replace('/', '.') + 'subprocess_main')
py_modules.append(prefix.replace('/', '.') + 'convert')
py_modules.append(prefix.replace('/', '.') + 'convert_impl')
py_modules.append(prefix.replace('/', '.') + '__main__')

class InstallCmd(install):
    def run(self):
        install.run(self)
        # copy requirements.txt files for all the frameworks
        for name in requirements_txt:
            copy(name, os.path.join(self.install_purelib, prefix))

        version_txt = 'version.txt'
        if os.path.exists(version_txt):
            copyfile(os.path.join(version_txt),
                     os.path.join(self.install_purelib, prefix, version_txt))


class BuildCmd(build_py):
    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        return [
            (pkg, module, filename)
            for (pkg, module, filename) in modules
        ]


packages = find_namespace_packages(prefix[:-1])
packages = [prefix.replace('/', '.') + p for p in packages]

setup(
    name='openvino-mo',
    version='0.0.0',
    author='Intel Corporation',
    author_email='openvino_pushbot@intel.com',
    url='https://github.com/openvinotoolkit/openvino',
    packages=packages,
    py_modules=py_modules,
    cmdclass={
        'install': InstallCmd,
        'build_py': BuildCmd,
    },
    entry_points={
        'console_scripts': [
            'mo = openvino.tools.mo.__main__:main',
        ],
    },
    package_data={
      'openvino.tools.mo.front.caffe.proto': ['*.proto'],
      'openvino.tools.mo.front.onnx': ['*.json'],
      'openvino.tools.mo.front.tf': ['*.json'],
      'openvino.tools.mo.front.caffe': ['CustomLayersMapping.xml*']
    },
    extras_require={
      'caffe': read_requirements('requirements_caffe.txt'),
      'kaldi': read_requirements('requirements_kaldi.txt'),
      'onnx': read_requirements('requirements_onnx.txt'),
      'tensorflow': read_requirements('requirements_tf.txt'),
      'tensorflow2': read_requirements('requirements_tf2.txt'),
    },
    classifiers=[
      "Programming Language :: Python :: 3",
      "License :: OSI Approved :: Apache Software License",
      "Operating System :: OS Independent",
    ],
    install_requires=read_requirements('requirements.txt'),
)
