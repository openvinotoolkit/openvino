#!/usr/bin/env python3

# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Use this script to create a wheel with OpenVINO™ Python* tools:

$ python setup.py sdist bdist_wheel
"""

from setuptools import setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open('benchmark/requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="openvino-tools",
    version="0.0.0",
    author='Intel® Corporation',
    author_email='openvino_pushbot@intel.com',
    url='https://github.com/openvinotoolkit/openvino',
    description="OpenVINO™ Python* tools package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    entry_points={
        'console_scripts': [
            'benchmark_app = openvino.tools.benchmark.main:main'],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={'openvino.tools.benchmark': 'benchmark',
                 'openvino.tools.benchmark.utils': 'benchmark/utils'},
    packages=['openvino.tools.benchmark', 'openvino.tools.benchmark.utils'],
    install_requires=required,
    python_requires=">=3.6",
)
