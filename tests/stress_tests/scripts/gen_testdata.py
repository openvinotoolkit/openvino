#!/usr/bin/env python3

# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

""" Script to generate XML cohfig file with the list of IR models
Usage: 
  python get_testdata.py  --test_conf <name_test_config>.xml  --ir_cache_dir <path_to_ir_cache>
or:
  python get_testdata.py  --test_conf <name_test_config>.xml --omz_cache_dir <path_to_ir_cache>
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
import xml.etree.ElementTree as eET

log.basicConfig(format="{file}: [ %(levelname)s ] %(message)s".format(file=os.path.basename(__file__)),
                level=log.INFO, stream=sys.stdout)


def abs_path(relative_path):
    """Return absolute path given path relative to the current file.
    """
    return os.path.realpath(
        os.path.join(os.path.dirname(getsourcefile(lambda: 0)), relative_path))


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

def get_args(parser):
    """Parse command line options
    """
    parser.add_argument('--test_conf', required=True, type=Path,
                        help='Path to a test config .xml file containing models '
                             'which will be downloaded and converted to IRs via OMZ.')
    parser.add_argument('--omz_repo', required=False,
                        help='Path to Open Model Zoo (OMZ) repository. It will be used to skip cloning step.')
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
    parser.add_argument('--ir_cache_dir', type=Path,
                        default=abs_path('../ir_cache'),
                        help='Directory with IR data cache.')
    return parser.parse_args()

def main():
    """Main entry point.
    """
    parser = argparse.ArgumentParser(description='Acquire test data',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = get_args(parser)
    test_conf_obj = ET.parse(str(args.test_conf))
    model_recs = get_model_recs(test_conf_obj.getroot()) # <class 'xml.etree.ElementTree.Element'>

    if not os.path.exists(args.ir_cache_dir) and not os.path.exists(args.omz_cache_dir):
        raise Exception("No \"ir_cache_dir\" or \"omz_cache_dir\" was not found")
    
    if os.path.exists(args.ir_cache_dir):
        subdirectory = str(args.ir_cache_dir)
    else:
        subdirectory = str(args.omz_cache_dir)

    for root, dirs, files in os.walk(subdirectory):
        for file in files:
            if file.endswith(".xml"):
                full_path = os.path.join(root, file)
                a = os.path.normpath(full_path).split(os.path.sep)
                aa = eET.Element("model")
                aa.attrib["name"] = a[-1]
                aa.tail = '\n\t'
                aa.attrib["precision"] = a[-4] if a[-2]!="optimized" else a[-5]
                aa.attrib["framework"] = a[-6] if a[-2]!="optimized" else a[-8]
                aa.attrib["path"] = subdirectory
                aa.attrib["full_path"] = full_path
                model_recs.append(aa)

    test_conf_obj.write(args.test_conf, xml_declaration=True)

if __name__ == "__main__":
    main()
