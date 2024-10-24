#!/usr/bin/env python3

# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Use this script to create a wheel with OpenVINO™ Python* tools:

$ python -m pip wheel .
"""
from setuptools import setup, find_packages


with open('requirements.txt', 'r', encoding='utf-8') as f:
    raw_contents = f.readlines()
    reqs = [line.strip() for line in raw_contents]

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

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
    install_requires=reqs,
    data_files=[('.', ['requirements.txt'])],
    python_requires='>=3.9',
)
