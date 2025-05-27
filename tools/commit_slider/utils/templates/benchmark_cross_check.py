from utils.templates.common_template import Template

class BenchmarkCrossCheckTemplate(Template):
    def getClassName():
        return 'bm_cc'
    
    def printResult(commitPath, outLogger, getCommitInfo):
        print("\n*****************<Commit slider output>*******************\nTable of throughputs:" + '\n\t\tm1\tm2')
        for pathcommit in commitPath.getList():
            from utils.common_mode import Mode
            commitInfo = getCommitInfo(pathcommit)
            cHash = pathcommit.cHash
            cHash = cHash[:7]
            commMsg = "{}\t\t{}\t{}".format(
                cHash,
                pathcommit.firstThroughput,
                pathcommit.secondThroughput)
            print(commMsg)
            outLogger.info(commitInfo)
        cPath = commitPath.getList()
        thr00 = float(cPath[0].firstThroughput)
        thr01 = float(cPath[0].secondThroughput)
        thr10 = float(cPath[1].firstThroughput)

        print("{}\nSupposed rootcause is: {}".format(
            "*" * 58,
            'Model' if abs(thr00 - thr01) > abs(thr00 - thr10) else 'OV'
        ))
        print('**********************************************************\n')
        print("*  m1 = {}".format(pathcommit.firstModel))
        print("** m2 = {}".format(pathcommit.secondModel))

    def generateFullConfig(cfg, customCfg):
        tmpJSON = cfg
        if '-v' in customCfg and customCfg['-v'] == 'false':
            tmpJSON['verboseOutput'] = False
        if '-c' in customCfg:
            # WA: additional args are passed with '-' with standart argparser
            customCfg['c'] = customCfg['-c']
        if 'c' in customCfg:
            curCfg = tmpJSON['runConfig']
            interval = customCfg['c'].split("..")
            start, end = interval[0], interval[1]
            curCfg['commitList'] = {'explicitList': [ start, end ] }
            tmpJSON['runConfig'] = curCfg
        # todo: move common method to helpers
        if 'gitPath' in customCfg:
            tmpJSON['gitPath'] = customCfg['gitPath']
        if 'buildPath' in customCfg:
            tmpJSON['buildPath'] = customCfg['buildPath']
        if 'appPath' in customCfg:
            tmpJSON['appPath'] = customCfg['buildPath']
        if 'appCmd' in customCfg:
            if isinstance(customCfg['appCmd'], list):
                curCfg = tmpJSON['runConfig']
                curCfg['firstAppCmd'] = customCfg['appCmd'][0]
                curCfg['secondAppCmd'] = customCfg['appCmd'][1]
                tmpJSON['runConfig'] = curCfg
            else:
                curCfg = tmpJSON['runConfig']
                curCfg['firstAppCmd'] = customCfg['appCmd'].format(model=customCfg['modelList'][0])
                curCfg['secondAppCmd'] = customCfg['appCmd'].format(model=customCfg['modelList'][1])
                tmpJSON['runConfig'] = curCfg
            tmpJSON['appCmd'] = "{actualAppCmd}"

        return tmpJSON