# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC
import utils.helpers as util
from utils.subscription import SubscriptionManager
from utils.break_validator import validateBMOutput
from utils.break_validator import BmValidationError
import json
import os
from enum import Enum
import csv

class Template(ABC):
    def getTemplateByCfg(cfg):
        return Template

    def printResult(commitPath, outLogger, getCommitInfo):
        print("*****************************\n* Broken compilation found: *\n*   <Template for output>   *\n*****************************\n")
        if not commitPath.metaInfo["preValidationPassed"]:
            msg = "Preliminary check failed, reason: {}".format(
                commitPath.metaInfo["reason"]
            )
            print(msg)
            outLogger.info(msg)
        elif not commitPath.metaInfo["postValidationPassed"]:
            msg = "Output results invalid, reason: {}".format(
                commitPath.metaInfo["reason"]
            )
            print(msg)
            outLogger.info(msg)
        else:
            for pathcommit in commitPath.getList():
                from utils.common_mode import Mode
                if pathcommit.state is not Mode.CommitPath.CommitState.DEFAULT:
                    commitInfo = getCommitInfo(pathcommit)
                    print(commitInfo)
                    outLogger.info(commitInfo)

    def prepareOutput():
        pass
    @staticmethod
    def getTemplate():
        from utils.templates.broken_compilation import BrokenCompilationTemplate
        return BrokenCompilationTemplate

    # @staticmethod
    # def factory(cfg):
    #     tmplClassName = util.checkAndGetClassnameByConfig(
    #         cfg, "modeMap", "mode"
    #     )
    #     keyName = cfg["runConfig"][specialCfg]
    #     map = cfg[mapName]
    #     if not (keyName in map):
    #         raise CfgError(
    #             "{keyName} is not registered in {mapName}".format(
    #                 keyName=keyName, mapName=mapName
    #             )
    #         )
    #     else:
    #         return map[keyName]
    #         cl = util.checkAndGetSubclass(modeClassName, Template)
    #     return cl(cfg)

    # def __init__(self, cfg) -> None:
    #     self.checkCfg(cfg)
    #     self.commitPath = self.CommitPath()
    #     traversalClassName = util.checkAndGetClassnameByConfig(
    #         cfg, "traversalMap", "traversal"
    #     )
    #     traversalClass = util.checkAndGetSubclass(
    #         traversalClassName, self.Traversal
    #     )
    #     self.traversal = traversalClass(self)
    #     self.cfg = cfg
    #     logPath = util.getActualPath("logPath", cfg)
    #     self.commonLogger = util.setupLogger(
    #         "commonLogger", logPath, "common_log.log"
    #     )
    #     self.outLogger = util.setupLogger(
    #         "outLogger", logPath, "out_log.log"
    #     )
    #     self.checkOutPathPattern = "check_output_cache.json"
