# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from os import walk, path
from utils.helpers import CfgError

def buildMap(commonPath, subPath=None) -> dict:
    precomPath = commonPath
    items = next(walk(precomPath), (None, None, []))[1]
    map = {}
    for item in items:
        item = item.replace('"', '')
        curPath = path.join(precomPath, item)
        if (subPath is not None and subPath):
            curPath = path.join(curPath, subPath)
        if path.exists(curPath):
            map[item] = curPath
    return map


def printMap(args):
    args = vars(args)
    if "-path" not in args:
        raise CfgError("No 'path' for map builder provided")
    if "-subPath" in args:
        print(buildMap(args["-path"], args["-subPath"]))
    else:
        print(buildMap(args["-path"]))
