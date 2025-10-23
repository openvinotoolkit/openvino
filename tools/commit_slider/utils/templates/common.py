# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC
from utils.subscription import SubscriptionManager
from utils.break_validator import BmValidationError
from utils.helpers import getClassByMethod
from enum import Enum
from utils.cfg_manager import CfgManager

class CommonTemplate(ABC):
    def getName():
        # simplified ID for API
        return 'common_template'
    
    def getBilletFileName():
        # filename for template billet to be defined by config population
        # 'some_custom_template.json' (".. {par_1} .. {par_N} ..") + parameters = full config,
        return None

    def getTemplateByCfg(cfg):
        if CommonTemplate.getName() == cfg['template']:
            return CommonTemplate
        else:
            return getClassByMethod('getName', cfg['template'], CommonTemplate)
    
    def passParameters(parList, srcCfg, dstCfg):
        for par in parList:
            if par in srcCfg:
                dstCfg[par] = srcCfg[par]

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

    @staticmethod
    def getTmplByMethod(target, method):
        def templateNameByFileName(filename):
            filename = filename.split('_')
            filename = ''.join(word.capitalize() for word in filename)
            return filename + 'Template'
        import os
        import importlib.util
        curDir = os.path.dirname(__file__)
        for filename in os.listdir(curDir):
            if filename.endswith(".py"):
                filename = filename[:-3]
                className = templateNameByFileName(filename)
                filename = 'utils.templates.' + filename
                module = importlib.import_module(filename)
                templateClass = getattr(module, className)
                if getattr(templateClass, method)() == target:
                    return templateClass
        return CommonTemplate
    
    @staticmethod
    def getTemplate(tmplName):
        return CommonTemplate.getTmplByMethod(tmplName, 'getName')

    @staticmethod
    def populateCfg(tmplName, cfg):
        tmpl = CommonTemplate.getTemplate(tmplName)
        cfgBillet = tmpl.getBilletFileName()
        fullCfg = CommonTemplate.getTemplate(tmplName).generateFullConfig(CfgManager.readJsonTmpl(cfgBillet), cfg)
        return fullCfg