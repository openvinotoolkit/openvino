# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from utils.map_builder import buildMap
from utils.e2e_preparator import buildWheelMap


class SubscriptionManager():
    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def apply(self):
        for sub in self.cfg["subscriptions"]:
            if not sub["enabled"]:
                continue
            else:
                subName = sub["name"]
                if subName == "wheelPathsMap":
                    self.generateWheelPathMap(self.cfg)
                elif subName == "wheelVersionsMap":
                    self.generateWheelVersionsMap(self.cfg)
                else:
                    raise Exception(
                        "No '{}' subscription registered".format(subName)
                        )

    def generateWheelPathMap(self, cfg: map):
        cashCfg = cfg["cachedPathConfig"]
        cashMap = buildMap(
            cashCfg["commonPath"], cashCfg["subPath"]
        )
        for k, v in cashMap.items():
            cfg["cachedPathConfig"]["cashMap"][k] = v
    
    def generateWheelVersionsMap(self, cfg: map):
        wheelMap = buildWheelMap(
            cfg["dlbConfig"]["commonPath"],
            cfg["dlbConfig"]["subPath"]
        )
        for k, v in wheelMap.items():
            cfg["dlbConfig"]["wheelVersionsMap"][k] = v