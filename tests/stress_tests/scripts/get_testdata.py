#!/usr/bin/env python3

# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

""" Script to acquire model IRs for stress tests.
Usage: ./scrips/get_testdata.py
"""
# pylint:disable=line-too-long

import argparse
import re
import logging as log
import os
import sys
from pathlib import Path
import glob
import xml.etree.ElementTree as ET
import string

log.basicConfig(format="{file}: [ %(levelname)s ] %(message)s".format(file=os.path.basename(__file__)),
                level=log.INFO, stream=sys.stdout)

datatype_re = re.compile(r"((FP|INT)\d+|(\d+BIT))")

def map_datatype(t: str) -> str:
    return {"4BIT": "INT4"}.get(t, t)


def testname_safestring(s: str) -> str:
    def translate_char(c:str) -> str:
        if c.isalnum():
            return c
        return '_'
    return ''.join(map(translate_char, s))


class ConfigFormatError(ValueError):
    pass


def read_test_config(test_conf_root):
    if test_conf_root is None:
        raise ConfigFormatError("no root element")
    models_element = test_conf_root.find("models")
    if models_element is None:
        raise ConfigFormatError("no 'models' element")
    for model in models_element.findall("model"):
        modelidpath = model.attrib.get("path")
        if modelidpath is None:
            raise ConfigFormatError("model has no 'path' attribute")
        datatypes = model.attrib.get("precision")
        yield (modelidpath, {
            "precision": tuple(datatypes.split("-")) if datatypes else None,
            "fullpath": model.attrib.get("fullpath")
        })


def main():
    parser = argparse.ArgumentParser(description='Acquire test data')
    parser.add_argument('--test_conf', '--test-conf', required=True, type=Path,
                        help='Path to a test config .xml file containing models '
                             'which will be downloaded and converted to IRs via OMZ')
    parser.add_argument('--ir_cache', '--ir-cache', type=Path, nargs='*', required=True,
                        help='Path to directory with *.xml model files, '
                        'each model will run, the last part of the path can be '
                        'a wildcard expression, if matches more that one directory, '
                        'selects the newest. Directories must have no intersections.')
    args = parser.parse_args()

    found_models: dict[str, dict] = {}

    for path in args.ir_cache:
        matching_dirs = sorted(
            glob.glob(str(path)),
            key=lambda item: os.lstat(item).st_ctime
        )
        if not matching_dirs:
            raise Exception("Path {path} not found")
        cache_dir = os.path.abspath(os.path.normpath(matching_dirs[-1]))
        log.info(f"Scanning requested {path} -> {cache_dir}")

        for modelfullpath in glob.glob(f"{cache_dir}/**/*.xml", recursive=True):
            modelidpath = modelfullpath.removeprefix(cache_dir)
            modelkey = testname_safestring(modelidpath)
            if modelkey in found_models:
                log.error(f"model key {modelkey} for {modelidpath} already exists: {found_models[modelkey]['path']}; skipping this one")
                continue
            found_models[modelkey] = {
                "precision": tuple([
                    map_datatype(m[0].upper()) 
                    for m in datatype_re.findall(modelidpath)
                ]),
                "full_path": modelfullpath,
                "path": modelidpath,
                "name": modelidpath.removesuffix(".xml")
            }

    test_conf_tree = ET.ElementTree(ET.fromstring("""
    <attributes>
        <devices>
            <value>CPU</value>
            <value>GPU</value>
        </devices>
        <models></models>
    </attributes>
    """))
    models_element = test_conf_tree.find("models")

    for modelid, model in found_models.items():
        model["precision"] = "-".join(model["precision"])
        models_element.append(ET.Element("model", model))

    test_conf_tree.write(args.test_conf)


if __name__ == "__main__":
    main()
