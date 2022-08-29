#!/usr/bin/env python3

# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Use this script to create a wheel with Model Optimizer code:

$ python setup.py sdist bdist_wheel
"""

import os
import re
import sys
from pathlib import Path
from shutil import copyfile, copy

from setuptools import setup, find_namespace_packages
from setuptools.command.build_py import build_py
from setuptools.command.install import install

prefix = 'openvino/tools/mo/'
SETUP_DIR = Path(__file__).resolve().parent / Path(prefix)


def read_text(path):
    return (Path(__file__).resolve().parent / path).read_text()


# Detect all the framework specific requirements_*.txt files.
requirements_txt = []
py_modules = []
for item in os.listdir():
    if re.match(r'requirements(.*)\.txt', item):
        requirements_txt.append(item)
for item in os.listdir(prefix):
    if re.match(r'mo(.*)\.py|main(.*)\.py', item):
        py_modules.append(prefix.replace('/', '.') + item.split('.')[0])
py_modules.append(prefix.replace('/', '.') + 'subprocess_main')
py_modules.append(prefix.replace('/', '.') + 'convert')
py_modules.append(prefix.replace('/', '.') + 'convert_impl')
py_modules.append(prefix.replace('/', '.') + '__main__')

# Minimal set of dependencies
deps_whitelist = ['networkx', 'defusedxml', 'numpy', 'openvino-telemetry']

deps = []
with open('requirements.txt', 'rt') as req_file:
    for line in req_file.read().split('\n'):
        if line.startswith(tuple(deps_whitelist)):
            deps.append(line)

# for py37 and less on Windows need importlib-metadata in order to use entry_point *.exe files
if sys.platform == 'win32' and sys.version_info[1] < 8:
    deps.append('importlib-metadata')


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
      'openvino.tools.mo.front.mxnet': ['*.json'],
      'openvino.tools.mo.front.onnx': ['*.json'],
      'openvino.tools.mo.front.tf': ['*.json'],
    },
    extras_require={
      'caffe': read_text('requirements_caffe.txt'),
      'kaldi': read_text('requirements_kaldi.txt'),
      'mxnet': read_text('requirements_mxnet.txt'),
      'onnx': read_text('requirements_onnx.txt'),
      'tensorflow': read_text('requirements_tf.txt'),
      'tensorflow2': read_text('requirements_tf2.txt'),
    },
    classifiers=[
      "Programming Language :: Python :: 3",
      "License :: OSI Approved :: Apache Software License",
      "Operating System :: OS Independent",
    ],
    install_requires=deps,
)
