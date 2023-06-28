# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys

from shutil import copyfile
from setuptools import setup, find_packages
from setuptools.command.install import install

UNKNOWN_VERSION = "unknown version"
here = os.path.abspath(os.path.dirname(__file__))
prefix = os.path.join("openvino", "tools")

with open(os.path.join(here, 'README.md'), 'r') as fh:
    long_description = fh.read()


class InstallCmd(install):
    def run(self):
        install.run(self)

        if self.root is None and self.record is None:
            # install requires
            self.do_egg_install()

        version_txt = os.path.join(prefix, "pot", "version.txt")
        if os.path.exists(version_txt):
            copyfile(os.path.join(version_txt),
                     os.path.join(self.install_purelib, version_txt))


def generate_pot_version():
    try:
        pot_dir = os.path.normpath(os.path.join(here, prefix))
        branch_name = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=pot_dir)
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=pot_dir)
        return "custom_{}_{}".format(branch_name.strip().decode(), commit_hash.strip().decode())
    except Exception: # pylint:disable=W0703
        return UNKNOWN_VERSION


def get_version():
    version_txt = os.path.join(here, prefix, "pot", "version.txt")
    if os.path.isfile(version_txt):
        with open(version_txt) as f:
            version = f.readline().replace('\n', '')
    else:
        version = generate_pot_version()
        with open(version_txt, 'w') as f:
            f.write(version + '\n')
    return version


INSTALL_EXTRAS = False
INSTALL_DEV_EXTRAS = False

if '--install-extras' in sys.argv:
    INSTALL_EXTRAS = True
    sys.argv.remove('--install-extras')

if '--install-dev-extras' in sys.argv:
    INSTALL_DEV_EXTRAS = True
    sys.argv.remove('--install-dev-extras')

INSTALL_REQUIRES = [
    "numpy>=1.16.6",
    "scipy~=1.7; python_version == '3.7'",
    "scipy>=1.8,<1.12; python_version >= '3.8'",
    "jstyleson>=0.0.2",
    "addict>=2.4.0",
    "networkx<=3.1",
    "tqdm>=4.54.1",
    "texttable>=1.6.3",
    "openvino-telemetry>=2022.1.0"
]

ALGO_EXTRAS = []

DEV_EXTRAS = [
    "pytest>=5.0,<=7.0.1;python_version<'3.10'",
    "pytest==7.2.0;python_version>='3.10'",
    "py>=1.9.0",
    "pytest-mock==3.1.1"
]

DEPENDENCY_LINKS = []

python_version = sys.version_info[:2]
if python_version[0] < 3:
    print('Only Python >= 3.7 is supported by POT/OV')
    sys.exit(0)
elif python_version[1] < 7:
    print('Only Python >= 3.7 is supported by POT/OV')
    sys.exit(0)


OS_POSTFIXES = {
    'win32': 'win_amd64',
    'linux': 'linux_x86_64',
}

version_string = "{}{}".format(sys.version_info[0], sys.version_info[1])
version_string_with_mem_manager = version_string + 'm' if sys.version_info[1] < 8 else version_string
os_string = None if sys.platform not in OS_POSTFIXES else OS_POSTFIXES[sys.platform]

TORCH_VERSION = '1.12.1'
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
    version='0.0.0',
    author='Intel',
    author_email='alexander.kozlov@intel.com',
    description='Post-training Optimization Toolkit',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://software.intel.com/openvino-toolkit',
    packages=find_packages(exclude=["tests", "tests.*",
                                    "tools", "tools.*"]),
    package_data={'openvino.tools.pot.configs.hardware': ['*.json'],
                  'openvino.tools.pot.api.samples': ['*.md', '*/*.md'],
                  'openvino.tools.pot.configs.templates': ['*.json']},
    include_package_data=True,
    cmdclass={
        'install': InstallCmd,
    },
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
