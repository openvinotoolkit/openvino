#!/usr/bin/env python3

# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Use this script to create a wheel with OpenVINO™ Cross Check Tool:

$ python setup.py sdist bdist_wheel
"""
from pathlib import Path
from setuptools import setup, find_packages

SETUP_DIR = Path(__file__).resolve().parent


def read_text(path):
    return (SETUP_DIR / path).read_text()

setup(
    name='cross_check_tool',
    version='0.0.0',
    author='Intel® Corporation',
    license='OSI Approved :: Apache Software License',
    author_email='openvino_pushbot@intel.com',
    url='https://github.com/openvinotoolkit/openvino',
    description='OpenVINO™ Cross Check Tool package',
    entry_points={
        'console_scripts': [
            'cross_check_tool = openvino.tools.cross_check_tool.cross_check_tool:main'],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    packages=find_packages(),
    install_requires=read_text('requirements.txt'),
    python_requires='>=3.6',
)
