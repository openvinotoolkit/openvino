#!/usr/bin/env python3

# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

""" Script to acquire model IRs for stress tests.
Usage: ./scrips/get_testdata.py
"""
# pylint:disable=line-too-long

import argparse
import json
import logging as log
import os
import shutil
import subprocess
import sys
from shutil import copytree
from inspect import getsourcefile
from pathlib import Path
import defusedxml.ElementTree as ET

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
            self.venv_executable = self.venv_dir / "Scripts" / "python.exe"

    def get_venv_executable(self):
        """Returns path to executable from virtual environment."""
        return str(self.venv_executable)

    def get_venv_dir(self):
        """Returns path to virtual environment root directory."""
        return str(self.venv_dir)

    def create(self):
        """Creates virtual environment."""
        cmd = '"{executable}" -m venv {venv}'.format(executable=sys.executable,
                                                     venv=self.get_venv_dir())
        run_in_subprocess(cmd)
        self.is_created = True

    def install_requirements(self, *requirements):
        """Installs provided requirements. Creates virtual environment if it hasn't been created."""
        if not self.is_created:
            self.create()
        cmd = '"{executable}" -m pip install --upgrade pip'.format(executable=self.get_venv_executable())
        for req in requirements:
            # Don't install requirements via one `pip install` call to prevent "ERROR: Double requirement given"
            cmd += ' && "{executable}" -m pip install -r {req}'.format(executable=self.get_venv_executable(), req=req)
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


def get_model_recs(test_conf_root):
    """Parse models from test config.
       Model records in multi-model configs with static test definition are members of "device" sections
    """
    device_recs = test_conf_root.findall("device")
    if device_recs:
        model_recs = []
        for device_rec in device_recs:
            for model_rec in device_rec.findall("model"):
                model_recs.append(model_rec)

        return model_recs

    return test_conf_root.find("models")


def main():
    """Main entry point.
    """
    parser = argparse.ArgumentParser(description='Acquire test data',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--test_conf', required=True, type=Path,
                        help='Path to a test config .xml file containing models '
                             'which will be downloaded and converted to IRs via OMZ.')
    parser.add_argument('--omz_repo', required=False,
                        help='Path to Open Model Zoo (OMZ) repository. It will be used to skip cloning step.')
    # TODO: update discretion for --mo_tool. When MO installed to python venv path to MO is redundant
    parser.add_argument('--mo_tool', type=Path,
                        help='Path to Model Optimizer (MO) runner. Required for OMZ converter.py only.')
    parser.add_argument('--omz_models_out_dir', type=Path,
                        default=abs_path('../_omz_out/models'),
                        help='Directory to put test data into. Required for OMZ downloader.py and converter.py.')
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

    # prepare Open Model Zoo
    if args.omz_repo:
        omz_path = Path(args.omz_repo).resolve()
    else:
        omz_path = Path(abs_path('..')) / "_open_model_zoo"
        # clone Open Model Zoo into temporary path
        if os.path.exists(str(omz_path)):
            shutil.rmtree(str(omz_path))
        cmd = 'git clone --single-branch --branch master' \
              ' https://github.com/openvinotoolkit/open_model_zoo {omz_path}'.format(omz_path=omz_path)
        run_in_subprocess(cmd)

    # prepare virtual environment and install requirements
    python_executable = sys.executable
    if not args.no_venv:
        # TODO: Update paths to requirements, currently paths are wrong
        Venv = VirtualEnv("./.stress_venv")
        requirements = [
            args.mo_tool.parents[3] / "requirements.txt",
            omz_path / "tools" / "model_tools" / "requirements.in",
            omz_path / "tools" / "model_tools" / "requirements-pytorch.in"
        ]
        Venv.create_n_install_requirements(*requirements)
        python_executable = Venv.get_venv_executable()

    test_conf_obj = ET.parse(str(args.test_conf))
    test_conf_root = test_conf_obj.getroot()
    model_recs = get_model_recs(test_conf_root)

    for model_rec in model_recs:
        if "name" not in model_rec.attrib or model_rec.attrib.get("source") != "omz":
            continue
        model_name = model_rec.attrib["name"]
        precision = model_rec.attrib["precision"]

        info_dumper_path = omz_path / "tools" / "model_tools" / "info_dumper.py"
        cmd = '"{executable}" "{info_dumper_path}" --name {model_name}'.format(executable=python_executable,
                                                                               info_dumper_path=info_dumper_path,
                                                                               model_name=model_name)
        try:
            out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True, universal_newlines=True)
        except subprocess.CalledProcessError as exc:
            log.warning(exc.output)
            continue

        model_info = json.loads(out)[0]

        # update model record from test config with Open Model Zoo info
        fields_to_add = ["framework", "subdirectory"]
        info_to_add = {key: model_info[key] for key in fields_to_add}
        # check selected precision with model info from Open Model Zoo
        if precision not in model_info['precisions']:
            log.warning("Please specify precision for the model "
                        f"{model_name} from the list: {model_info['precisions']}")
            model_recs.remove(model_rec)
            continue
        model_rec.attrib.update(info_to_add)
        model_rec.attrib["path"] = str(
            Path(model_rec.attrib["subdirectory"]) / precision / (model_rec.attrib["name"] + ".xml"))
        model_rec.attrib["full_path"] = str(
            args.omz_irs_out_dir / model_rec.attrib["subdirectory"] / precision / (model_rec.attrib["name"] + ".xml"))

        # prepare models

        downloader_path = omz_path / "tools" / "model_tools" / "downloader.py"
        cmd = '"{executable}" {downloader_path} --name {model_name}' \
              ' --precisions={precision}' \
              ' --num_attempts {num_attempts}' \
              ' --output_dir {models_dir}' \
              ' --cache_dir {cache_dir}'.format(executable=python_executable, downloader_path=downloader_path,
                                                model_name=model_name,
                                                precision=precision, num_attempts=OMZ_NUM_ATTEMPTS,
                                                models_dir=args.omz_models_out_dir, cache_dir=args.omz_cache_dir)
        run_in_subprocess(cmd, check_call=not args.skip_omz_errors)

        # convert models to IRs
        converter_path = omz_path / "tools" / "model_tools" / "converter.py"

        # NOTE: remove --precisions if both precisions (FP32 & FP16) required
        cmd = f'"{python_executable}" {converter_path} --name {model_name}' \
              f' -p "{python_executable}"' \
              f' --precisions={precision}' \
              f' --output_dir {args.omz_irs_out_dir}' \
              f' --download_dir {args.omz_models_out_dir}'
        if args.mo_tool:
            cmd += f' --mo {args.mo_tool}'
        run_in_subprocess(cmd, check_call=not args.skip_omz_errors)

    for model_rec in model_recs:
        if model_rec.attrib.get("full_path") is None:
            log.warning(f"Model {model_rec.attrib['name']} does not have 'full_path' attribute! "
                        f"This model will not be verified in this run.")
            model_recs.remove(model_rec)

    # rewrite test config with updated records
    test_conf_obj.write(args.test_conf)

    # Open Model Zoo doesn't copy downloaded IRs to converter.py output folder where IRs should be stored.
    # Do it manually to have only one folder with IRs
    for ir_src_path in args.omz_models_out_dir.rglob("*.xml"):
        ir_dst_path = args.omz_irs_out_dir / os.path.relpath(ir_src_path, args.omz_models_out_dir)
        # allows copy to an existing folder
        copytree(str(ir_src_path.parent), str(ir_dst_path.parent))


if __name__ == "__main__":
    main()
