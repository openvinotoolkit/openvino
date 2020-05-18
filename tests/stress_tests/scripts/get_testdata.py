#!/usr/bin/env python3
# Copyright (C) 2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


""" Script to acquire model IRs for stress tests.
Usage: ./scrips/get_testdata.py
"""
# pylint:disable=line-too-long

import argparse
import logging as log
import multiprocessing
import os
import shutil
import subprocess
import sys
from inspect import getsourcefile
from pathlib import Path
from xml.etree import ElementTree as ET

log.basicConfig(format="{file}: [ %(levelname)s ] %(message)s".format(file=os.path.basename(__file__)),
                level=log.INFO, stream=sys.stdout)

# Parameters
OMZ_NUM_ATTEMPTS = 6


def abs_path(relative_path):
    """Return absolute path given path relative to the current file.
    """
    return os.path.realpath(
        os.path.join(os.path.dirname(getsourcefile(lambda: 0)), relative_path))


class VirtualEnv:
    """Class implemented creation and use of virtual environment."""
    is_created = False

    def __init__(self, venv_dir):
        self.venv_dir = Path(abs_path('..')) / venv_dir
        if sys.platform.startswith('linux') or sys.platform == 'darwin':
            self.venv_executable = self.venv_dir / "bin" / "python3"
        else:
            self.venv_executable = self.venv_dir / "Scripts" / "python3.exe"

    def get_venv_executable(self):
        """Returns path to executable from virtual environment."""
        return str(self.venv_executable)

    def get_venv_dir(self):
        """Returns path to virtual environment root directory."""
        return str(self.venv_dir)

    def create(self):
        """Creates virtual environment."""
        cmd = '{executable} -m venv {venv}'.format(executable=sys.executable,
                                                   venv=self.get_venv_dir())
        run_in_subprocess(cmd)
        self.is_created = True

    def install_requirements(self, *requirements):
        """Installs provided requirements. Creates virtual environment if it hasn't been created."""
        if not self.is_created:
            self.create()
        cmd = '{executable} -m pip install --upgrade pip'.format(executable=self.get_venv_executable())
        for req in requirements:
            # Don't install requirements via one `pip install` call to prevent "ERROR: Double requirement given"
            cmd += ' && {executable} -m pip install -r {req}'.format(executable=self.get_venv_executable(), req=req)
        run_in_subprocess(cmd)

    def create_n_install_requirements(self, *requirements):
        """Creates virtual environment and installs provided requirements in it."""
        self.create()
        self.install_requirements(*requirements)


def run_in_subprocess(cmd, check_call=True):
    """Runs provided command in attached subprocess."""
    log.info(cmd)
    if check_call:
        subprocess.check_call(cmd, shell=True)
    else:
        subprocess.call(cmd, shell=True)


def main():
    """Main entry point.
    """
    parser = argparse.ArgumentParser(
        description='Acquire test data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--test_conf', required=True, type=Path,
                        help='Path to a test config .xml file containing models '
                             'which will be downloaded and converted to IRs via OMZ.')
    parser.add_argument('--omz_repo', required=False,
                        help='Path to Open Model Zoo (OMZ) repository. It will be used to skip cloning step.')
    parser.add_argument('--mo_tool', type=Path,
                        default=Path(abs_path('../../../model-optimizer/mo.py')).resolve(),
                        help='Path to Model Optimizer (MO) runner. Required for OMZ converter.py only.')
    parser.add_argument('--omz_models_out_dir', type=Path,
                        default=abs_path('../_omz_out/models'),
                        help='Directory to put test data into. Required for OMZ downloader.py and converter.py')
    parser.add_argument('--omz_irs_out_dir', type=Path,
                        default=abs_path('../_omz_out/irs'),
                        help='Directory to put test data into. Required for OMZ converter.py only.')
    parser.add_argument('--omz_cache_dir', type=Path,
                        default=abs_path('../_omz_out/cache'),
                        help='Directory with test data cache. Required for OMZ downloader.py only.')
    parser.add_argument('--no_venv', action="store_true",
                        help='Skip preparation and use of virtual environment to convert models via OMZ converter.py.')
    parser.add_argument('--skip_omz_errors', action="store_true",
                        help='Skip errors caused by OMZ while downloading and converting.')
    args = parser.parse_args()

    # Step 0: prepare models list
    tree = ET.parse(str(args.test_conf))
    root = tree.getroot()
    models_names = []
    for attributes in root:
        if attributes.tag == "models":
            models = [child.text for child in attributes]
            models_names = [Path(model).stem for model in models]
            break

    os.makedirs(str(args.omz_models_out_dir), exist_ok=True)
    models_list_path = args.omz_models_out_dir / "models_list.txt"
    log.info("List of models from {models_list_path} used for downloader.py and converter.py: "
             "{models_names}".format(models_list_path=models_list_path, models_names=",".join(models_names)))
    with open(str(models_list_path), "w") as file:
        file.writelines([name + "\n" for name in models_names])

    # Step 1: prepare Open Model Zoo
    if args.omz_repo:
        omz_path = Path(args.omz_repo).resolve()
    else:
        omz_path = Path(abs_path('..')) / "_open_model_zoo"
        # Clone Open Model Zoo into temporary path
        if os.path.exists(str(omz_path)):
            shutil.rmtree(str(omz_path))
        cmd = 'git clone https://github.com/opencv/open_model_zoo {omz_path}'.format(omz_path=omz_path)
        run_in_subprocess(cmd)

    # Step 3: prepare models
    downloader_path = omz_path / "tools" / "downloader" / "downloader.py"
    cmd = '{downloader_path} --list {models_list_path}' \
          ' --num_attempts {num_attempts}' \
          ' --output_dir {models_dir}' \
          ' --cache_dir {cache_dir}'.format(downloader_path=downloader_path, models_list_path=models_list_path,
                                            num_attempts=OMZ_NUM_ATTEMPTS,
                                            models_dir=args.omz_models_out_dir,
                                            cache_dir=args.omz_cache_dir)
    run_in_subprocess(cmd, check_call=not args.skip_omz_errors)

    # Step 4: prepare virtual environment and install requirements
    python_executable = sys.executable
    if not args.no_venv:
        Venv = VirtualEnv("./.stress_venv")
        requirements = [
            omz_path / "tools" / "downloader" / "requirements.in",
            args.mo_tool.parent / "requirements.txt",
            args.mo_tool.parent / "requirements_dev.txt",
            omz_path / "tools" / "downloader" / "requirements-caffe2.in",
            omz_path / "tools" / "downloader" / "requirements-pytorch.in"
        ]
        Venv.create_n_install_requirements(*requirements)
        python_executable = Venv.get_venv_executable()

    # Step 5: convert models to IRs
    converter_path = omz_path / "tools" / "downloader" / "converter.py"
    # NOTE: remove --precision if both precisions (FP32 & FP16) required
    cmd = '{executable} {converter_path} --list {models_list_path}' \
          ' -p {executable}' \
          ' --precision=FP32' \
          ' --output_dir {irs_dir}' \
          ' --download_dir {models_dir}' \
          ' --mo {mo_tool} --jobs {workers_num}'.format(executable=python_executable, converter_path=converter_path,
                                                        models_list_path=models_list_path, irs_dir=args.omz_irs_out_dir,
                                                        models_dir=args.omz_models_out_dir, mo_tool=args.mo_tool,
                                                        workers_num=multiprocessing.cpu_count())
    run_in_subprocess(cmd, check_call=not args.skip_omz_errors)


if __name__ == "__main__":
    main()
