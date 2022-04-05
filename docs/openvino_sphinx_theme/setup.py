# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from setuptools import setup

setup(
    name='openvino-sphinx-theme',
    version='0.0.1',
    packages=['openvino_sphinx_theme'],
    maintainer='Nikolay Tyukaev',
    maintainer_email='nikolay.tyukaev@intel.com',
    include_package_data=True,
    entry_points={"sphinx.html_themes": ["openvino_sphinx_theme = openvino_sphinx_theme"]},
    install_requires=['pydata_sphinx_theme', 'sphinx_inline_tabs'],
    python_requires='>=3.5',
    url='https://github.com/openvinotoolkit/openvino',
    license='',
    author='Intel Corporation',
    author_email='',
    description='A Sphinx theme',
    long_description=''
)
