# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import os


class CfgManager():
    def __init__(self, cfg) -> None:
        self.cfg = cfg

    @staticmethod
    def singlestepStrFormat(input: str, placeholder: str, substitution: str):
        return input.replace(
            '{}{}{}'.format('{', placeholder, '}'),
            substitution
        )

    @staticmethod
    def multistepStrFormat(input: str, placeholderSubstPairArr):
        for ps in placeholderSubstPairArr:
            input = CfgManager.singlestepStrFormat(input, ps['p'], ps['s'])
        return input

    def applyTemplate(self):
        if not "template" in self.cfg:
            return self.cfg
        tmplName = self.cfg["template"]["name"]
        fullCfg = {}
        # todo: generalize tmplcfg generator
        if tmplName == "bm_simple":
            fullCfg = self.generatebmSimpleTemplate()
        elif tmplName == "e2e":
            fullCfg = self.generateE2ETemplate()
        elif tmplName == "bm_functional":
            fullCfg = self.generatebmFunctionalTemplate()
        elif tmplName == "bm_arm_mac":
            fullCfg = self.generateArmBmTemplate()
        else:
            raise Exception(
                "Unknown template '{}'".format(tmplName)
            )
        return fullCfg
    
    def readJsonTmpl(self, tmplFileName: str):
        smplPath = os.path.dirname(os.path.realpath(__file__))
        tmplFileName = os.path.join(
            smplPath, "cfg_samples/", tmplFileName
        )
        with open(tmplFileName) as cfgFile:
            tmplJSON = json.load(cfgFile)
            return tmplJSON

    def generateE2ETemplate(self):
        tmpl = self.cfg["template"]
        tmpJSON = self.readJsonTmpl("e2e_for_CI.json")

        if "errorPattern" in tmpl and\
            "precommitPath" in tmpl and\
            "testCmd" in tmpl:
            tmpJSON["runConfig"]["stopPattern"] = tmpl["errorPattern"]
            tmpJSON["dlbConfig"]["commonPath"] = tmpl["precommitPath"]
            tmpJSON["cachedPathConfig"]["commonPath"] = tmpl["precommitPath"]
            tmpJSON["appCmd"] = CfgManager.singlestepStrFormat(
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

    def generatebmFunctionalTemplate(self):
        tmpl = self.cfg["template"]
        tmpJSON = self.readJsonTmpl("bm_output.json")
        stopPattern = "stopPattern"
        if "appCmd" in tmpl:
            tmpJSON["appCmd"] = tmpl["appCmd"]
        else:
            raise("No 'appcmd' in template")
        if stopPattern in tmpl:
            tmpJSON["runConfig"][stopPattern] = tmpl[stopPattern]
        else:
            raise("No 'stopPattern' in template")
        return tmpJSON

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

    def generateArmBmTemplate(self):
        tmpl = self.cfg["template"]
        tmpJSON = self.readJsonTmpl("bm_arm_mac.json")
        # handle syntactic sugar: logs, models, intervals
        isMultimodel = False
        if isinstance(tmpl['model'], list):
            isMultimodel = True
            tmpJSON['clearCache'] = True
            appCmdList = []
            logPathList = []
            for model in tmpl['model']:
                appCmdList.append(
                    CfgManager.singlestepStrFormat(
                        tmpJSON['appCmd'], 'model', model
                    )
                )
                logPathList.append(
                    CfgManager.singlestepStrFormat(
                        "{workPath}/log/sublog/{model}",
                        'model',
                        model
                        ))
            tmpJSON['appCmd'] = appCmdList
        if 'interval' in tmpl:
            runCfgList = []
            for interval in tmpl['interval']:
                runCfg = tmpJSON['runConfig']
                cmtList = "\"commitList\" : {\
                    \"getCommitListCmd\" : \"git log {interval} \
                    --boundary --pretty=\\\"%h\\\"\"}"
                CfgManager.singlestepStrFormat(
                        cmtList, 'interval', interval)
                runCfg['commitList'] = cmtList
                runCfgList.append(runCfg)
            tmpJSON['runConfig'] = runCfgList

        devParam = "perfAppropriateDeviation"
        isFirstFixed = "isFirstFixed"
        if isMultimodel:
            pass
        elif "appCmd" in tmpl:
            tmpJSON["appCmd"] = tmpl["appCmd"]
        elif 'model' in tmpl: # check if model param separated
            newCmd = CfgManager.singlestepStrFormat(
                tmpJSON['appCmd'], "model", tmpl['model']
            )
            tmpJSON['appCmd'] = newCmd
        else:
            raise Exception("No 'appcmd' in template")
        if devParam in tmpl:
            tmpJSON["runConfig"][devParam] = tmpl[devParam]
        if isFirstFixed in tmpl and tmpl[isFirstFixed]:
            tmpJSON["runConfig"]["traversal"] = "firstFixedVersion"
        else:
            tmpJSON["runConfig"]["traversal"] = "firstFailedVersion"
        return tmpJSON

# Example 1
# {
#    "template": {
#       "name":"bm_arm_mac",
#       "buildEnvVars" : [
#          {"name" : "ALL_PROXY", "val" : "<path_1>"},
#          {"name" : "http_proxy", "val" : "<path_2>"},
#          {"name" : "https_proxy", "val" : "<path_3>"}
#       ],
#       "model":["path_0", "path_1", "path_2"]
#    }
# }

# Example 2
# {
#    "template": {
#       "name":"bm_arm_mac",
#       "buildEnvVars" : [
#          {"name" : "ALL_PROXY", "val" : "<path_1>"},
#          {"name" : "http_proxy", "val" : "<path_2>"},
#          {"name" : "https_proxy", "val" : "<path_3>"}
#       ],
#       "appCmd":"./benchmark_app -m {model} <other_parameters>",
#       "model":["path_0", "path_1", "path_2"],
#       "traversal":"firstFixedVersion",
#       "perfAppropriateDeviation": 0.2,
#       "interval":["hash_00..hash_01", "hash_10..hash_11", "hash_20..hash_21"]
#    }
# }

# Example 3
# {
#    "template": {
#       "name":"bm_arm_mac",
#       "buildEnvVars" : [
#          {"name" : "ALL_PROXY", "val" : "<path_1>"},
#          {"name" : "http_proxy", "val" : "<path_2>"},
#          {"name" : "https_proxy", "val" : "<path_3>"}
#       ],
#       "appCmd":[
#           "./benchmark_app -m <model_path_0> <other_parameters_0>",
#           "./benchmark_app -m <model_path_1> <other_parameters_1>",
#           "./benchmark_app -m <model_path_2> <other_parameters_2>",
#        ]
#    }
# }
