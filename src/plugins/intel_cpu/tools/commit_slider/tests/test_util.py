# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import re
import json
import sys
import shutil
from os import path
from test_data import TestData
from test_data import TestError

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
    for patch in patchList:
        state = 'EMPTY'
        if 'state' in patch:
            state = patch['state']
        sub = patch['str']
        newData = data[:prefixPos] + sub + "\n" + data[postfixPos + 1:]
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
        td,
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


def formatJSON(content, td: TestData, formatLambda):
    if isinstance(content, dict):
        for k, value in content.items():
            content[k] = formatJSON(value, td, formatLambda)
    elif isinstance(content, list):
        for id, item in enumerate(content):
            content[id] = formatJSON(item, td, formatLambda)
    elif isinstance(content, str):
        content = formatLambda(content)
    else:
        # bool or digit object
        pass
    return content


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


def getExpectedCommit(td: TestData):
    markedVersionList = createRepo(td)
    breakCommit = markedVersionList[[
        i for i in range(
            len(markedVersionList)
        ) if markedVersionList[i]['state'] == 'BREAK'][0]]['commit']
    breakCommit = getMeaningfullCommitTail(breakCommit)

    td.fillActualData(markedVersionList)
    td.actualDataReceived = True

    return breakCommit, td

def getActualCommit(td: TestData):
    if not td.actualDataReceived:
        raise TestError("Running actual commit before expected.")

    # prepare config
    cfg = formatJSON(td.testCfg, td, td.formatConfig)
    testCfg = "test_cfg.json"

    with open(testCfg, "w+") as customCfg:
        customCfg.truncate(0)
        json.dump(cfg, customCfg)
    customCfg.close()

    # run slider and check output
    sliderOutput = runCmd(
        "python3.8 commit_slider.py -cfg tests/{}".format(testCfg),
        "../")

    sliderOutput = '\n'.join(map(str, sliderOutput))
    foundCommit = re.search(
            "Break commit: (.*),", sliderOutput, flags=re.MULTILINE
        ).group(1)

    # clear temp data
    [shutil.rmtree(dir) for dir in [
            td.repoName,
            td.cachePath,
            td.logPath]]
    os.remove(testCfg)

    return foundCommit