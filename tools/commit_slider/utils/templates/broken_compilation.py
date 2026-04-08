from utils.templates.common import CommonTemplate

class BrokenCompilationTemplate(CommonTemplate):
    def getName():
        return 'broken_comp'

    def getBilletFileName():
        return 'broken_compilation.json'
    
    def printResult(commitPath, outLogger, getCommitInfo):
        print("\n*****************<Commit slider output>*******************\n* Commit with broken compilation found:" + ' ' * 18 + '*')
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
                    commMsg = "* {}".format(commitInfo)
                    print(commMsg + ' ' * (57 - len(commMsg)) + '*')
                    outLogger.info(commitInfo)
        print('**********************************************************\n')

    def generateFullConfig(cfg, customCfg):
        tmpJSON = cfg
        if '-v' in customCfg and customCfg['-v'] == 'false':
            tmpJSON['verboseOutput'] = False
        if '-c' in customCfg:
            # WA: additional args are passed with '-' with standart argparser
            customCfg['c'] = customCfg['-c']
        if 'c' in customCfg:
            curCfg = tmpJSON['runConfig']
            curCfg['commitList'] = {'getCommitListCmd': "git log {} --boundary --pretty=\"%h\"".format(customCfg['c'])}
            tmpJSON['runConfig'] = curCfg
        # todo: move common method to helpers
        if 'gitPath' in customCfg:
            tmpJSON['gitPath'] = customCfg['gitPath']
        if 'buildPath' in customCfg:
            tmpJSON['buildPath'] = customCfg['buildPath']
            tmpJSON['appPath'] = customCfg['buildPath']

        return tmpJSON