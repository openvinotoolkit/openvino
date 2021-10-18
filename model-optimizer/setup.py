#!/usr/bin/env python3

# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Use this script to create a wheel with Model Optimizer code:

$ python setup.py sdist bdist_wheel
"""

import os
import re
from pathlib import Path
from shutil import copyfile

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
py_modules.append(prefix.replace('/', '.') + '__main__')

# Minimal set of dependencies
deps_whitelist = ('networkx', 'defusedxml', 'numpy')
deps = []
with open('requirements.txt', 'rt') as req_file:
    for line in req_file.read().split('\n'):
        if line.startswith(deps_whitelist):
            deps.append(line)


class InstallCmd(install):
    def run(self):
        install.run(self)
        # Create requirements.txt files for all the frameworks
        for name in requirements_txt:
            path = os.path.join(self.install_purelib, prefix, name)
            with open(path, 'wt') as common_reqs_file:
                common_reqs_file.write('\n'.join(deps))
        # Add version.txt if exists
        version_txt = 'version.txt'
        if os.path.exists(version_txt):
            copyfile(os.path.join(version_txt),
                     os.path.join(self.install_purelib, prefix, version_txt))

        path = os.path.join(self.install_purelib, prefix, '__init__.py')
        with open(path, 'wt') as init_file:
            init_file.write('import os, sys\n')
            init_file.write('from openvino.tools.mo import mo \n')
            # This is required to fix internal imports
            init_file.write('sys.path.append(os.path.dirname(__file__))\n')
            # Redirect import to model-optimizer/mo/__init__.py
            init_file.write('sys.modules["mo"] = mo')


class BuildCmd(build_py):
    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        return [
            (pkg, module, filename)
            for (pkg, module, filename) in modules
        ]


packages = find_namespace_packages(prefix)
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
            'summarize_graph = openvino.tools.mo.utils.summarize_graph:main'
        ],
    },
    package_data={
      'mo.mo.front.caffe.proto': ['*.proto'],
      'mo.extensions.front.mxnet': ['*.json'],
      'mo.extensions.front.onnx': ['*.json'],
      'mo.extensions.front.tf': ['*.json'],
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