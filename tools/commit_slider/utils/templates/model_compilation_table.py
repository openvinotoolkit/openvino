from utils.templates.common import CommonTemplate

class ModelCompilationTableTemplate(CommonTemplate):
    def getName():
        return 'model_comp_table'

    def getBilletFileName():
        return 'model_compilation_time_table.json'
    
    def printResult(commitPath, outLogger, getCommitInfo):
        print("\n*****************<Commit slider output>*******************\nTable of compilation time:" + '\n\t\tm1\tm2')
        for pathcommit in commitPath.getList():
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
        else:
            tmpJSON['verboseOutput'] = True
        if '-c' in customCfg:
            # WA: additional args are passed with '-' with standart argparser
            customCfg['c'] = customCfg['-c']
        if 'c' in customCfg:
            curCfg = tmpJSON['runConfig']
            interval = customCfg['c'].split("..")
            start, end = interval[0], interval[1]
            curCfg['commitList'] = {'explicitList': [ start, end ] }
            tmpJSON['runConfig'] = curCfg
        CommonTemplate.passParameters([
            'gitPath', 'buildPath', 'appPath'
            ], customCfg, tmpJSON)
        if 'appCmd' in customCfg:
            if isinstance(customCfg['appCmd'], list):
                curCfg = tmpJSON['runConfig']
                curCfg['par_1'] = customCfg['appCmd'][0]
                curCfg['par_2'] = customCfg['appCmd'][1]
                tmpJSON['runConfig'] = curCfg
            else:
                curCfg = tmpJSON['runConfig']
                curCfg['par_1'] = customCfg['appCmd'].format(model=customCfg['modelList'][0])
                curCfg['par_2'] = customCfg['appCmd'].format(model=customCfg['modelList'][1])
                tmpJSON['runConfig'] = curCfg
            tmpJSON['appCmd'] = "{actualPar}"

        return tmpJSON