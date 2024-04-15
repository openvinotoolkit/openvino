# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import os


class CfgManager():
    def __init__(self, cfg) -> None:
        self.cfg = cfg

    @staticmethod
    def multistepStrFormat(input: str, placeholder: str, substitution: str):
        return input.replace(
            '{}{}{}'.format('{', placeholder, '}'),
            substitution
        )

    def applyTemplate(self):
        if not "template" in self.cfg:
            return self.cfg
        logPath = self.cfg["logPath"]
        tmplName = self.cfg["template"]["name"]
        fullCfg = {}
        if tmplName == "bm_simple":
            fullCfg = self.generatebmSimpleTemplate()
        elif tmplName == "e2e":
            fullCfg = self.generateE2ETemplate()
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
        if "appCmd" in tmpl:
            tmpJSON["appCmd"] = tmpl["appCmd"]
        else:
            raise("No 'appcmd' in template")
        if devParam in tmpl:
            tmpJSON["runConfig"][devParam] = tmpl[devParam]
        return tmpJSON

def generateE2ETemplate(self):
        tmpl = self.cfg["template"]
        tmpJSON = self.readJsonTmpl("e2e_for_CI.json")

        if "errorPattern" in tmpl and\
            "precommitPath" in tmpl and\
            "testCmd" in tmpl:
            tmpJSON["runConfig"]["stopPattern"] = tmpl["errorPattern"]
            tmpJSON["dlbConfig"]["commonPath"] = tmpl["precommitPath"]
            tmpJSON["cachedPathConfig"]["commonPath"] = tmpl["precommitPath"]
            tmpJSON["appCmd"] = CfgManager.multistepStrFormat(
                tmpJSON["appCmd"],
                tmpl["appCmd"],
                "testCmd"
            )
        else:
            raise("Template is incomplete.")
        subPath = "private_linux_manylinux2014_release/"
        if "subPath" in tmpl:
            subPath = tmpl["subPath"]
        tmpJSON["dlbConfig"]["subPath"] = subPath
        tmpJSON["cachedPathConfig"]["subPath"] = subPath

        return tmpJSON
