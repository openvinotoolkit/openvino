# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC
import utils.helpers as util
from utils.subscription import SubscriptionManager
from utils.break_validator import validateBMOutput
from utils.break_validator import BmValidationError
from utils.helpers import getClassByMethod
import json
import os
from enum import Enum
import csv

class Template(ABC):
    def getClassName():
        # to redefine in subclasses,
        # is used for class fabric
        return 'common_template'

    def getTemplateByCfg(cfg):
        if Template.getClassName() == cfg['template']:
            return Template
        else:
            return getClassByMethod('getClassName', cfg['template'], Template)

    def printResult(commitPath, outLogger, getCommitInfo):
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
    def getTemplate(tmplName):
        # # WA: automatic import cannot find all modules
        # # todo: investigate
        # import importlib
        # curDir = os.path.dirname(__file__)
        # for filename in os.listdir(curDir):
        #     if filename.endswith(".py"):
        #         modulename = filename[:-3]
        #         importlib.import_module(modulename)

        if tmplName == 'broken_compilation':
            from utils.templates.broken_compilation import BrokenCompilationTemplate
            return BrokenCompilationTemplate
        elif tmplName == 'bm_cc':
            from utils.templates.benchmark_cross_check import BenchmarkCrossCheckTemplate
            return BenchmarkCrossCheckTemplate
        else:
            return Template
