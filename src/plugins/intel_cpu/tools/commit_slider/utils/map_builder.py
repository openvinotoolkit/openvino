from os import walk, path
import json
from utils.helpers import CfgError

def makeMap(commonPath, subPath=None):
    precomPath = commonPath
    items = next(walk(precomPath), (None, None, []))[1]
    map = {}
    for item in items:
        item = item.replace('"', '')
        curPath = path.join(precomPath, item)
        if subPath is not None:
            curPath = path.join(curPath, subPath)
        map[item] = curPath
    print(json.dumps(map))


def map_builder(args):
    if "-path" not in args:
        raise CfgError("No 'path' for map builder provided")
    if "-subPath" in args:
        makeMap(args["-path"], args["-subPath"])
    else:
        makeMap(args["-path"])
