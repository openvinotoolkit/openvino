# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import os


class CfgManager():
    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def applyTemplate(self):
        if not "template" in self.cfg:
            return self.cfg
        logPath = self.cfg["logPath"]
        tmplName = self.cfg["template"]["name"]
        fullCfg = {}
        if tmplName == "bm_simple":
            fullCfg = self.generatebmSimpleTemplate()
        else:
            raise Exception(
                "Unknown template '{}'".format(tmplName)
            )
        fullCfg["logPath"] = logPath
        return fullCfg
    
    def readJsonTmpl(self, tmplFileName: str):
        tmplFileName = os.path.join(
            "utils/cfg_samples/", tmplFileName
        )
        with open(tmplFileName) as cfgFile:
            tmplJSON = json.load(cfgFile)
            return tmplJSON

    def generatebmSimpleTemplate(self):
        tmpl = self.cfg["template"]
        tmpJSON = self.readJsonTmpl("bm_perf_for_CI.json")
        devParam = "perfAppropriateDeviation"
        isFirstFixed = "isFirstFixed"
        if "appCmd" in tmpl:
            tmpJSON["appCmd"] = tmpl["appCmd"]
        else:
            raise("No 'appcmd' in template")
        if devParam in tmpl:
            tmpJSON["runConfig"][devParam] = tmpl[devParam]
        if isFirstFixed in tmpl and tmpl[isFirstFixed]:
            tmpJSON["runConfig"]["traversal"] = "firstFixedVersion"
        else:
            tmpJSON["runConfig"]["traversal"] = "firstFailedVersion"
        return tmpJSON
