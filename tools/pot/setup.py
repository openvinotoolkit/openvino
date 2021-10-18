# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import re

import codecs
from setuptools import setup, find_packages


here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, 'README.md'), 'r') as fh:
    long_description = fh.read()


def read(*parts):
    with codecs.open(os.path.join(here, *parts), 'r') as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError('Unable to find version string.')


INSTALL_EXTRAS = False
INSTALL_DEV_EXTRAS = False

if '--install-extras' in sys.argv:
    INSTALL_EXTRAS = True
    sys.argv.remove('--install-extras')

if '--install-dev-extras' in sys.argv:
    INSTALL_DEV_EXTRAS = True
    sys.argv.remove('--install-dev-extras')

INSTALL_REQUIRES = [
    'scipy~=1.5.4',
    'jstyleson~=0.0.2',
    'numpy>=1.16.6,<1.20',
    'addict>=2.4.0',
    'networkx~=2.5',
    'tqdm>=4.54.1',
    'texttable~=1.6.3',
    'pandas~=1.1.5',
]

ALGO_EXTRAS = [
    'hyperopt~=0.1.2',
]

DEV_EXTRAS = ['pytest==4.5.0', 'openpyxl==2.6.4', 'pytest-mock==3.1.1']

DEPENDENCY_LINKS = []

python_version = sys.version_info[:2]
if python_version[0] < 3:
    print('Only Python >= 3.6 is supported by POT/OV')
    sys.exit(0)
elif python_version[1] < 6:
    print('Only Python >= 3.6 is supported by POT/OV')
    sys.exit(0)


OS_POSTFIXES = {
    'win32': 'win_amd64',
    'linux': 'linux_x86_64',
}

version_string = "{}{}".format(sys.version_info[0], sys.version_info[1])
version_string_with_mem_manager = version_string + 'm' if sys.version_info[1] < 8 else version_string
os_string = None if sys.platform not in OS_POSTFIXES else OS_POSTFIXES[sys.platform]

TORCH_VERSION = '1.8.1'
TORCH_SOURCE_URL_TEMPLATE = 'https://download.pytorch.org/whl/cpu/torch-{tv}%2Bcpu-cp{ver}-cp{' \
                            'ver_m}-{os}.whl'

torch_source_url = TORCH_SOURCE_URL_TEMPLATE.format(
        tv=TORCH_VERSION,
        ver=version_string,
        ver_m=version_string_with_mem_manager,
        os=os_string,
    )

torch_dependency = 'torch @ {}'.format(torch_source_url) \
    if os_string is not None else 'torch=={}'.format(TORCH_VERSION)

ALGO_EXTRAS.extend([torch_dependency])

if INSTALL_EXTRAS:
    INSTALL_REQUIRES.extend(ALGO_EXTRAS)
if INSTALL_DEV_EXTRAS:
    INSTALL_REQUIRES.extend(DEV_EXTRAS)

DEPENDENCY_LINKS = [torch_source_url]


setup(
    name='pot',
    version=find_version(os.path.join(here, 'openvino/tools/pot/version.py')),
    author='Intel',
    author_email='alexander.kozlov@intel.com',
    description='Post-training Optimization Toolkit',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://software.intel.com/openvino-toolkit',
    packages=find_packages(),
    package_data={"openvino.tools.pot.configs.hardware": ['*.json'],
                  "openvino.tools.pot.api.samples": ['*.md', '*/*.md']},
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: EULA for the Intel(R) Software Development Products',
        'Operating System :: OS Independent',
    ],
    install_requires=INSTALL_REQUIRES,
    dependency_links=DEPENDENCY_LINKS,
    entry_points={
        'console_scripts': [
            'pot=openvino.tools.pot.app.run:main'
        ]
    }
)
