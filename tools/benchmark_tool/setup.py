#!/usr/bin/env python3

# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Use this script to create a wheel with OpenVINO™ Python* tools:

$ python setup.py sdist bdist_wheel
"""
import pkg_resources
import re
from setuptools import setup, find_packages
from pathlib import Path
from typing import Dict, List


with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()


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


setup(
    name='benchmark_tool',
    version='0.0.0',
    author='Intel® Corporation',
    license='OSI Approved :: Apache Software License',
    author_email='openvino_pushbot@intel.com',
    url='https://github.com/openvinotoolkit/openvino',
    description='OpenVINO™ Python* tools package',
    long_description=long_description,
    long_description_content_type='text/markdown',
    entry_points={
        'console_scripts': [
            'benchmark_app = openvino.tools.benchmark.main:main'],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    packages=find_packages(),
    install_requires=read_requirements('requirements.txt'),
    python_requires='>=3.8',
)
