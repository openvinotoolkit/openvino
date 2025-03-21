# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import re
import json
import sys
import shutil
from os import path
from utils.cfg_manager import CfgManager
from test_data import TestData
from test_data import TestError
from utils.helpers import formatJSON

sys.path.append('../')
from utils.helpers import getMeaningfullCommitTail

def getVersionList(td: TestData):
    with open(td.patchedFile, 'r') as file:
        data = file.read()

    # extract patch list
    stats_re = re.compile(td.pattern, re.MULTILINE | re.DOTALL)
    patchJSONList = stats_re.findall(data)
    if not patchJSONList:
        raise Exception("Wrong patchlist in {}".format(td.patchedFile))

    patchJSONList = patchJSONList[0]
    patchList = json.loads(patchJSONList)
    prefixPos = re.search(td.patternPrefix, data, re.DOTALL).span()[0]
    postfixPos = re.search(td.patternPostfix, data, re.DOTALL).span()[1]

    # apply patches and fill version list
    versionList = []
    for id, patch in enumerate(patchList):
        state = 'EMPTY'
        if 'state' in patch:
            state = patch['state']
        sub = patch['str']
        newData = "// {}\n{}{}\n{}".format(
            id,  # distinctive string to guarantee that versions differ
            data[:prefixPos],
            sub, # patch
            data[postfixPos + 1:]
        )
        newVersion = {
            "content": newData,
            "state": state,
            "comment": patch['comment']}
        versionList.append(newVersion)
    return versionList


def makeRepoContent(td: TestData):
    fullPath = path.join(td.repoPath, td.repoName)

    td.repoStructure['files'] = formatJSON(
        td.repoStructure['files'],
        lambda content: content.format(
            repoName=td.repoName,
            mainFile=td.mainFile)
    )
    for file in td.repoStructure['files']:
        filePath = path.join(fullPath, file['name'])
        with open(filePath, "w") as textFile:
            textFile.write(file['content'])

    for dir in td.repoStructure['dirs']:
        dir = path.join(fullPath, dir)
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.makedirs(dir)

def runCmd(cmd, cwd, verbose=False):
    if verbose:
        print("run command: {}".format(cmd))

    proc = subprocess.Popen(
            cmd.split(),
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",errors="replace"
        )

    output = []
    for line in proc.stdout:
        if verbose:
            sys.stdout.write(line)
        output.append(line)
    proc.wait()
    proc.communicate()
    return output

def createRepo(td: TestData):
    repoName = td.repoName
    repoPath = td.repoPath
    repoPath = os.path.abspath(repoPath)
    runCmd("mkdir {}".format(repoName), repoPath)
    innerPath = path.join(repoPath, repoName)
    runCmd("git init", innerPath)

    makeRepoContent(td)

    for file in td.repoStructure['files']:
        cmd = "git add {}".format(file['name'])
        runCmd(cmd, innerPath)

    return commitPatchList(getVersionList(td), innerPath, td.mainFile)

def commitPatchList(versionList, innerPath, fileName):
    markedVersionList = []

    for version in versionList:
        with open(path.join(innerPath, fileName), "w") as textFile:
            textFile.write(version['content'])

        runCmd("git add {}".format(fileName), innerPath)
        runCmd("git commit -m \"{}\"".format(version['comment']), innerPath)
        hash = runCmd("git rev-parse HEAD", innerPath)[0]

        markedVersion = {
            "commit": hash.strip(),
            "state": version['state'],
            "comment": version['comment']
        }
        markedVersionList.append(markedVersion)

    return markedVersionList

def getBordersByTestData(td: TestData):
    return\
        getMeaningfullCommitTail(td.start),\
        getMeaningfullCommitTail(td.end)

def getExpectedCommit(td: TestData):
    markedVersionList = createRepo(td)
    breakList = [i for i in range(len(markedVersionList))\
                if markedVersionList[i]['state'] == 'BREAK']
    if not breakList:
        breakCommit = None
    else:
        breakCommit = markedVersionList[breakList[0]]['commit']
        breakCommit = getMeaningfullCommitTail(breakCommit)

    td.fillActualData(markedVersionList)
    td.actualDataReceived = True

    return breakCommit, td

def getActualCommit(td: TestData):
    sliderOutput = runCS(td)
    rejectReason, foundCommit = parseSliderOutput(sliderOutput)
    # clear temp data
    [shutil.rmtree(dir, ignore_errors=True) for dir in [
            td.repoName,
            td.userCachePath,
            td.userLogPath
            ]]
    os.remove(td.testCfgName)

    return foundCommit, rejectReason

def getCSOutput(td: TestData):
    sliderOutput = runCS(td)
    # clear temp data
    [shutil.rmtree(dir) for dir in [
            td.repoName,
            td.userCachePath,
            td.userLogPath
            ]]
    os.remove(td.testCfgName)

    return sliderOutput

def runCS(td: TestData):
    if not td.actualDataReceived:
        raise TestError("Running actual commit before expected.")

    # prepare config
    cfg = formatJSON(td.testCfg, td.formatConfig)
    td.userLogPath = CfgManager.singlestepStrFormat(
        td.userLogPath, "testDir", os.getcwd())
    td.userCachePath = CfgManager.singlestepStrFormat(
        td.userCachePath, "testDir", os.getcwd())
    td.testCfgName = "test_cfg.json"
    for key in [
        'userLogPath', 'clearLogsAposteriori',
        'userCachePath', 'clearCache'
            ]:
        if isinstance(cfg, list):
            cfg[0][key] = getattr(td, key)
        else:
            cfg[key] = getattr(td, key)
    with open(td.testCfgName, "w+") as customCfg:
        customCfg.truncate(0)
        json.dump(cfg, customCfg)
    customCfg.close()

    # run slider and check output
    sliderOutput = runCmd(
        "python3 commit_slider.py -cfg tests/{}".format(td.testCfgName),
        "../")

    sliderOutput = '\n'.join(map(str, sliderOutput))
    return sliderOutput

def parseSliderOutput(sliderOutput: str):
    rejectReason, foundCommit, matcher = None, None, None
    pattern = "Preliminary check failed, reason: (.*)"
    matcher = re.search(
            pattern, sliderOutput, flags=re.MULTILINE
        )
    if matcher is not None:
        rejectReason = matcher.group(1)
        return rejectReason, None

    pattern = "Break commit: (.*), state"
    matcher = re.search(
            pattern, sliderOutput, flags=re.MULTILINE
        )
    if matcher is not None:
        foundCommit = matcher.group(1)
        return None, foundCommit

    pattern = "Output results invalid, reason: (.*)"
    matcher = re.search(
            pattern, sliderOutput, flags=re.MULTILINE
        )
    if matcher is not None:
        rejectReason = matcher.group(1)
        return rejectReason, None
    else:
        raise TestError(
            "Unexpected output: {}".format(
                sliderOutput
            ))


def requireBinarySearchData(td: TestData, rsc: map):
    td.requireTestData(
            lambda td, rsc: setattr(td,
                'commonRsc',
                rsc['binarySearchRes'])
        )
    td.requireTestData(
        lambda td, rsc: [setattr(td, key, rsc[td.getTestName()][key]) for key in [
            'testCfg', 'patchedFile', 'repoName'
        ]]
    )
    [setattr(td, key, td.commonRsc[key] \
            if not key in td.testCfg or \
            not (isinstance(td.testCfg[key], str) or \
                 isinstance(td.testCfg[key], bool)) \
            else td.testCfg[key]) for key in [
        'repoStructure',
        'userCachePath',
        'userLogPath',
        'clearLogsAposteriori',
        'clearCache',
        'mainFile', 'repoPath'
    ]]
    td.patternPrefix = td.commonRsc['patchGeneratorPrefix']
    td.patternPostfix = td.commonRsc['patchGeneratorPostfix']
    td.pattern = "{pre}(.+?){post}".format(
        pre=td.patternPrefix,
        post=td.patternPostfix)
