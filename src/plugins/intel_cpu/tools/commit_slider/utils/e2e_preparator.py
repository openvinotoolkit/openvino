# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from os import walk, path
import json
from utils.helpers import CfgError
import yaml
from pathlib import Path

def getWheelVersion(manifestPath: str):
    fullPath = path.join(manifestPath, 'manifest.yml')
    try:
        conf = yaml.safe_load(Path(fullPath).read_text())
    except FileNotFoundError:
        return None
    return conf["components"]["dldt"]["custom_params"]["wheel_product_version"]

def buildMap(commonPath, subPath=None):
    # subPath specify system configuration. e.g. 'private_linux_ubuntu_18_04_release/'
    precomPath = commonPath
    items = next(walk(precomPath), (None, None, []))[1]
    map = {}
    for item in items:
        item = item.replace('"', '')
        curPath = path.join(precomPath, item)
        if (subPath is not None and subPath):
            curPath = path.join(curPath, subPath)
        wheelVersion = getWheelVersion(curPath)
        if wheelVersion is not None:
            map[item] = wheelVersion
    return json.dumps(map)


def printMap(args):
    args = vars(args)
    if "-path" not in args:
        raise CfgError("No 'path' for map builder provided")
    if "-subPath" not in args:
        raise CfgError("No 'subPath' for map builder provided")
    else:
        print(buildMap(args["-path"], args["-subPath"]))
