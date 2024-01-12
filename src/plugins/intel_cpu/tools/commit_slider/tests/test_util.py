import os
import subprocess
import re
import json
import sys
import shutil
from os import path

sys.path.append('../')
from utils.helpers import getMeaningfullCommitTail

def getVersionList(caseName, rsc):
    patternPrefix = rsc['CommonRes']['patchGeneratorPrefix']
    patternPostfix = rsc['CommonRes']['patchGeneratorPostfix']
    pattern = "{pre}(.+?){post}".format(pre=patternPrefix, post=patternPostfix)

    with open(rsc[caseName]['patchedFile'], 'r') as file:
        data = file.read()

    stats_re = re.compile(pattern, re.MULTILINE | re.DOTALL)
    patchJSONList = stats_re.findall(data)
    if not patchJSONList:
        raise Exception("Wrong patchlist in main.cpp")

    patchJSONList = patchJSONList[0]
    patchList = json.loads(patchJSONList)
    prefixPos = re.search(patternPrefix, data, re.DOTALL).span()[0]
    postfixPos = re.search(patternPostfix, data, re.DOTALL).span()[1]
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


def makeRepoContent(repoPath, repoName, rsc):
    fullPath = path.join(repoPath, repoName)
    cmakeLists = rsc['CommonRes']['cmakeTemplate']
    cmakeLists = cmakeLists.format(repoName=repoName)

    filePath = path.join(fullPath, "CMakeLists.txt")
    with open(filePath, "w") as text_file:
        text_file.write(cmakeLists)

    filePath = path.join(fullPath, ".gitignore")
    with open(filePath, "w") as text_file:
        text_file.write("/build\n")

    filePath = path.join(fullPath, "main.cpp")
    with open(filePath, "w") as text_file:
        text_file.write("")

    dir = path.join(fullPath, "build")
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)

def runCmd(cmd, cwd, verbose=True):
    if verbose:
        print("run command: {}".format(cmd))
    proc = subprocess.Popen(
            cmd.split(),
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8", errors="replace"
        )
    output = []
    for line in proc.stdout:
        if verbose:
            sys.stdout.write(line)
        output.append(line)
    proc.wait()
    proc.communicate()
    return output


def formatJSON(content, markedVersionList):
    if isinstance(content, dict):
        for k, value in content.items():
            content[k] = formatJSON(value, markedVersionList)
    elif isinstance(content, list):
        for id, item in enumerate(content):
            content[id] = formatJSON(item, markedVersionList)
    elif isinstance(content, str):
        # todo: load from test
        content = content.format(
            appCmd="./testRepo",
            appPath="tests/testRepo/build",
            buildPath="tests/testRepo/build",
            gitPath="tests/testRepo",
            start=markedVersionList[0]['commit'],
            end=markedVersionList[-1]['commit']
        )
    else:
        # bool or digit object
        pass
    return content


def createRepo(caseName, repoPath, repoName, rsc):
    repoPath = os.path.abspath(repoPath)
    cmd = "mkdir {}".format(repoName)
    runCmd(cmd, repoPath)
    innerPath = path.join(repoPath, repoName)
    cmd = "git init"
    runCmd(cmd, innerPath)
    makeRepoContent(repoPath, repoName, rsc)
    cmd = "git add CMakeLists.txt .gitignore main.cpp"
    runCmd(cmd, innerPath)
    return commitPatchList(getVersionList(caseName, rsc), innerPath, "main.cpp")

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

def checkTestCase(caseName):
    rsc = {}
    with open("tests_res.json") as cfgFile:
        rsc = json.load(cfgFile)
    cfgFile.close()

    markedVersionList = createRepo(caseName, "./", "testRepo", rsc)
    breakCommit = markedVersionList[[
        i for i in range(
            len(markedVersionList)
        ) if markedVersionList[i]['state'] == 'BREAK'][0]]['commit']
    with open("tests_res.json") as cfgFile:
        rsc = json.load(cfgFile)
        cfg = rsc[caseName]["cfg"]
        cfg = formatJSON(cfg, markedVersionList)
    cfgFile.close()
    with open("test_cfg.json", "w+") as customCfg:
        customCfg.truncate(0)
        json.dump(cfg, customCfg)
    customCfg.close()
    # run slider
    sliderOutput = runCmd(
        "python3.8 commit_slider.py -cfg tests/test_cfg.json",
        "../")
    sliderOutput = '\n'.join(map(str, sliderOutput))
    breakCommit = getMeaningfullCommitTail(breakCommit)
    foundCommit = re.search(
            "Break commit: (.*),", sliderOutput, flags=re.MULTILINE
        ).group(1)
    shutil.rmtree("testRepo")
    shutil.rmtree("../slider_cache")
    shutil.rmtree("../log")
    os.remove("test_cfg.json")
    return breakCommit == foundCommit
