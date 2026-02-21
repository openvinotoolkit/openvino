from utils.templates.common import CommonTemplate

class BenchmarkOvcTemplate(CommonTemplate):
    def getName():
        return 'bm_ovc'

    def getBilletFileName():
        return 'benchmark_ovc.json'
    
    def printResult(commitPath, outLogger, getCommitInfo):
        print("\n*****************<Commit slider output>*******************\n")
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
                if str(pathcommit.state).strip() != 'CommitState.DEFAULT':
                    # todo: add check 'break' and 'ignored' commits adjacency
                    # and move CommitState class for common use
                    commitInfo = getCommitInfo(pathcommit)
                    outLogger.info(commitInfo)
                    print(commitInfo)
        print('**********************************************************\n')

    def generateFullConfig(cfg, customCfg):
        tmpJSON = cfg
        if '-v' in customCfg and customCfg['-v'] == 'false':
            tmpJSON['verboseOutput'] = False
        if '-c' in customCfg:
            # WA: additional args are passed with '-' with standart argparser
            customCfg['c'] = customCfg['-c']
        if 'dev' in customCfg:
            curCfg = cfg['runConfig']
            curCfg['perfAppropriateDeviation'] = float(customCfg['dev'])
            tmpJSON['runConfig'] = curCfg
        if 'convert_command' in customCfg:
            curCfg = cfg['runConfig']
            curCfg['convertCommand'] = customCfg['convert_command']
            tmpJSON['runConfig'] = curCfg
        else:
            raise Exception("No 'convert_command' parameter passed in " + str(customCfg))
        if 'bm_command' in customCfg:
            curCfg = cfg['runConfig']
            curCfg['bmCommand'] = customCfg['bm_command']
            tmpJSON['runConfig'] = curCfg
        else:
            raise Exception("No 'bm_command' parameter passed in " + str(customCfg))
        if 'commit_path' in customCfg:
            curCfg = cfg['dlbConfig']
            curCfg['commonPath'] = customCfg['commit_path']
            tmpJSON['dlbConfig'] = curCfg
            curCfg = cfg['cachedPathConfig']
            curCfg['commonPath'] = customCfg['commit_path']
            tmpJSON['cachedPathConfig'] = curCfg
        else:
            raise Exception("No 'commit_path' parameter passed in " + str(customCfg))
        if 'sub_path' in customCfg:
            curCfg = cfg['dlbConfig']
            curCfg['subPath'] = customCfg['sub_path']
            tmpJSON['dlbConfig'] = curCfg
            curCfg = cfg['cachedPathConfig']
            curCfg['subPath'] = customCfg['sub_path']
            tmpJSON['cachedPathConfig'] = curCfg
        else:
            raise Exception("No 'sub_path' parameter passed in " + str(customCfg))

        return tmpJSON