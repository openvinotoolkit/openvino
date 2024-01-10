import os
import subprocess
import re
import json
import sys
import shutil
from os import path

sys.path.append('../')
from utils.helpers import getMeaningfullCommitTail

def getVersionList():
    patternPrefix = "const char \*patchGenerator = R\"V0G0N\("
    patternPostfix = "\)V0G0N\";"
    pattern = "{pre}(.+?){post}".format(pre=patternPrefix, post=patternPostfix)

    with open('main.cpp', 'r') as file:
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
        mark = 'EMPTY'
        if 'mark' in patch:
            mark = patch['mark']
        sub = patch['str']
        newData = data[:prefixPos] + sub + "\n" + data[postfixPos + 1:]
        newVersion = {"content": newData, "mark": mark, "comment": patch['comment']}
        versionList.append(newVersion)
    return versionList


def makeRepoContent(repoPath, repoName):
    fullPath = path.join(repoPath, repoName)
    cmakeLists = '''cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project({repoName} VERSION 1.0)
add_executable("${{PROJECT_NAME}}" "main.cpp")
'''.format(repoName=repoName)

    filePath = path.join(fullPath, "CMakeLists.txt")
    with open(filePath, "w") as text_file:
        text_file.write(cmakeLists)

    filePath = path.join(fullPath, ".gitignore")
    with open(filePath, "w") as text_file:
        text_file.write("/build\n")

    filePath = path.join(fullPath, "main.cpp")
    with open(filePath, "w") as text_file:
        text_file.write("")

    os.mkdir(path.join(fullPath, "build"))

def runCmd(cmd, cwd, verbose=False):
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

def createRepo(repoPath, repoName):
    repoPath = os.path.abspath(repoPath)
    cmd = "mkdir {}".format(repoName)
    runCmd(cmd, repoPath)
    innerPath = path.join(repoPath, repoName)
    cmd = "git init"
    runCmd(cmd, innerPath)
    makeRepoContent(repoPath, repoName)
    cmd = "git add CMakeLists.txt .gitignore main.cpp"
    runCmd(cmd, innerPath)
    return commitPatchList(getVersionList(), innerPath, "main.cpp")

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
            "mark": version['mark'],
            "comment": version['comment']
        }
        markedVersionList.append(markedVersion)
    return markedVersionList

def checkTestCase():
    rsc = {}
    with open("tests_res.json") as cfgFile:
        rsc = json.load(cfgFile)
    cfgFile.close()

    markedVersionList = createRepo("./", "testRepo")
    breakCommit = markedVersionList[[
        i for i in range(
            len(markedVersionList)
        ) if markedVersionList[i]['mark'] == 'BREAK'][0]]['commit']
    with open("tests_res.json") as cfgFile:
        rsc = json.load(cfgFile)
        cfg = rsc["FirstBadVersionTestRes"]["cfg"]
        repl = cfg["appCmd"]
        repl = repl.format(appCmd="./testRepo")
        cfg["appCmd"] = repl
        repl = cfg["appPath"]
        repl = repl.format(appPath="tests/testRepo/build")
        cfg["appPath"] = repl
        repl = cfg["buildPath"]
        repl = repl.format(buildPath="tests/testRepo/build")
        cfg["buildPath"] = repl
        repl = cfg["gitPath"]
        repl = repl.format(gitPath="tests/testRepo")
        cfg["gitPath"] = repl
        repl = cfg["runConfig"]
        gitCmd = repl["commitList"]["getCommitListCmd"]
        gitCmd = gitCmd.format(
            start=markedVersionList[0]['commit'],
            end=markedVersionList[-1]['commit']
        )
        repl["commitList"] = {"getCommitListCmd" : gitCmd}
        cfg["runConfig"] = repl
    cfgFile.close()
    with open("test_cfg.json", "w+") as customCfg:
        customCfg.truncate(0)
        json.dump(cfg, customCfg)
    customCfg.close()
    # run slider
    sliderOutput = runCmd(
        "python3.8 commit_slider.py -cfg tests/test_cfg.json",
        "../")[0]

    breakCommit = getMeaningfullCommitTail(breakCommit)
    foundCommit = re.search(
            "Break commit: (.*),", sliderOutput, flags=re.MULTILINE
        ).group(1)
    # print(breakCommit==foundCommit)
    shutil.rmtree("testRepo")
    shutil.rmtree("../slider_cache")
    shutil.rmtree("../log")
    os.remove("test_cfg.json")
    return breakCommit == foundCommit
