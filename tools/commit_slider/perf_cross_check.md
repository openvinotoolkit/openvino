# Case #1. Cross-checking of performance degradation.

([Basic concepts.](README.md) [More about templates.](common_architecture.md))

#### Themes: Mode and Template customization, API design
  

For 2 given versions, defined by commit hashes and 2 models we need to resolve which of two changes (ov or model) caused degradation. In other words, we need to differentiate 2 possible cases:

model-caused:
|'Bad' model | model_1.xml| model_2.xml|
| --------     | -------    | -------|
| ov_version_1 | 100 % (ref)|50%     |
| ov_version_2 | 98 %       |51%     |

and ov - caused

|'Bad' ov-version| model_1.xml| model_2.xml|
| --------     | -------    | -------|
| ov_version_1 | 100 % (ref)|98%     |
| ov_version_2 | 48% %       |51%     |

In the next steps we will use progressively more advanced methods to solve given problem.

## Version #1. Reusing of existing classes

### Traversal
We need full bypass of commit set (pair of commits in the given case), independently on the result of a specific commit.

It corresponds to the existing `brute_force` traversal.

### Mode

On every commit we run several commands.

For the first version, we also reuse the existing `Nop` (No Operation) mode with simple list of bash commands.

### Configuration
The simplest configuration with default settings (openvino/, build/ and bin/intel64/Release/ paths correspondingly).
```
We use explicitList instead of commit interval.
{
    "appCmd" : [
        "./benchmark_app -m model_1_path.xml",
        "./benchmark_app -m model_2_path.xml"
        ],
    "runConfig" : {
        "commitList" : {
            "explicitList" : [ "<start_commit>", "<end_commit>" ]
        },
    "mode" : "nop",
    "traversal" : "bruteForce"
}
```
Or specified with custom paths:
```
{
    "appCmd" : [
        "echo first model",
        "./benchmark_app -m model_1_path.xml",
        "echo second model",
        "./benchmark_app -m model_2_path.xml"
        ],
    "appPath": "<path_to_benchmark_app>",
    "gitPath" : "<path_to_git_repository>",
    "buildPath" : "<path_to_build_dir>",
    "runConfig" : {
        "commitList" : {
            "explicitList" : [ "<start_commit>", "<end_commit>" ]
    },
    "mode" : "nop",
    "traversal" : "bruteForce"
}
```
### Example of launch and output
```
python3 commit_slider.py -cfg cfg.json

current commit: <start_commit_hash>
first model
<...> Throughput: 500 FPS
second model
<...> Throughput: 500 FPS

current commit: <end_commit_hash>
first model
<...> Throughput: 1000 FPS
second model
<...> Throughput: 1000 FPS

```
It faces given problem, but looks clumsy and verbose. We also need to interpret results ourselves.

## Version #2: Specific Mode

If there is no Mode [Details about Mode](README.md# implementing-custom-traversal-rule) which resolves commit with our task or we need more readable output and opportunity to handle every commit directly, we can implement custom Mode. We can also pass specific parameters directly to Mode.
### Mode
```
# Mode supposes to be as much generic as possible
# and corresponds all similar problems,
# and not connected with the separate user case,
# but as the example we provide simplified version.
class  CrossCheckMode(Mode):
    def  __init__(self, cfg):
        super().__init__(cfg)

def  checkCfg(self, cfg):
    # preparation pattern for parsing and parameters
    self.firstAppCmd = cfg["runConfig"]["firstAppCmd"]
    self.secondAppCmd = cfg["runConfig"]["secondAppCmd"]
    self.outPattern = r'{spec}:\s*([0-9]*[.][0-9]*)\s*{measure}'.format(spec='Throughput', measure='FPS')
    super().checkCfg(cfg)

def  getPseudoMetric(self, commit, cfg):
    # main Mode method initially resolving two compared
    # commits (for performance goals it is numeric metric, as
    # soon as in some Modes 'metric' has non-numeric value,
    # e.g. filename we got pseudometric), for noncomparative
    # Modes it doesn't return any value, just run procedures
    # for handling single commit.
    commit = commit.replace('"', "")
    commitLogger = getCommitLogger(cfg, commit)
    self.commonLogger.info("New commit: {commit}".format(commit=commit))

    # building and compilation of given version
    handleCommit(commit, cfg)
    fullOutput = ""

    # passing each of the commandlines for execution
    simpleSubstitute(cfg, "actualAppCmd", "$.runConfig.firstAppCmd", "$.appCmd")

    # run first app
    checkOut = fetchAppOutput(cfg, commit)
    commitLogger.info(checkOut)
    foundThroughput = re.search(
        self.outPattern, checkOut, flags=re.MULTILINE
    ).group(1)

    # parsing of output and holding it with full commandline
    # and specific model for the final result
    self.firstThroughput = foundThroughput
    self.firstModel = cfg['appCmd']
    fullOutput = checkOut
    simpleSubstitute(cfg, "actualAppCmd", "$.runConfig.secondAppCmd", "$.appCmd")

    # run second app
    checkOut = fetchAppOutput(cfg, commit)
    commitLogger.info(checkOut)
    foundThroughput = re.search(
        self.outPattern, checkOut, flags=re.MULTILINE
    ).group(1)
    self.secondThroughput = foundThroughput
    self.secondModel = cfg['appCmd']

    # add commit data to commit path
    pc = Mode.CommitPath.PathCommit(
        commit,
        Mode.CommitPath.CommitState.DEFAULT
    )
    self.setOutputInfo(pc)
    self.commitPath.accept(self.traversal, pc)

def  printResult(self):
    # formatting of full output
    for  pathcommit  in  self.commitPath.getList():
        commitInfo = self.getCommitInfo(pathcommit)
        print(commitInfo)
        self.outLogger.info(commitInfo)

def  setOutputInfo(self, pathCommit):
    # populating of commit path
    pathCommit.firstThroughput = self.firstThroughput
    pathCommit.firstModel = self.firstModel
    pathCommit.secondThroughput = self.secondThroughput
    pathCommit.secondModel = self.secondModel

def  getCommitInfo(self, commit):
    return  "{hash}, throughput_1 = {t1}, throughput_2 = {t2}".format(
        hash=commit.cHash,
        t1=commit.firstThroughput,
        t2=commit.secondThroughput)
```
### Configuration

Supposedly we pass both command lines launching models in two parameters `cfg["runConfig"]["firstAppCmd"]` and `cfg["runConfig"]["secondAppCmd"]`, i.e.

```
cfg = { 
    "appCmd" : "{actualAppCmd}",
    "appPath": "<path_to_benchmark_app>",
    "gitPath" : "<path_to_git_repository>",
    "buildPath" : "<path_to_build_dir>",
        "runConfig" : {
            "commitList" : {
                "explicitList" : [ "{start}", "{end}" ]
            },
            "mode" : "crossCheck",
            "firstAppCmd" : "./benchmark_app -m model_1.xml --other_params",
            "secondAppCmd" : "./benchmark_app -m model_2.xml --other_params",
            "traversal" : "bruteForce"
        }
}
```
### Output
```
python3 commit_slider.py -cfg cfg.json

<hash_1>, throughput_1 = 500.0, throughput_2 = 500.0
<hash_2>, throughput_1 = 1000.0, throughput_2 = 1000.0

```
We've got rid of full output and learned to provide only necessary information, but we still limited by verbose configuration-style API and still provide raw output information.

## Version #3: Specific Template
###  API design
We want to provide for user 2 ways to set up command lines for comparation. ```'appCmd' : ['./benchmark_app --parameters_1', './benchmark_app --parameters_2']``` for detailed parameters and simplified ```'par_1' : 'parameter_1', 'par_2' : 'parameter_2'``` supposing that substitute the placeholder ```'appCmd' : 'prefix {actualPar} postfix'```. The example of this approach may be ```'par_1' : 'model_1.xml', 'par_2' : 'model_2.xml' : 'parameter_2', 'appCmd' : './benchmark_app -m {actualPar} -i input.png'```
  
### Custom template
```
from  utils.templates.common  import  CommonTemplate

class  BenchmarkCrossCheckTemplate(CommonTemplate):

def  printResult(commitPath, outLogger, getCommitInfo):
    # formatting of output anf post analisys
    print("\n" + "*" * 17 "<Commit slider output>" + "*" * 17 "\nTable of throughputs:" + '\n\t\thr_1\thr_2')
    for  pathcommit  in  commitPath.getList():
        commitInfo = getCommitInfo(pathcommit)
        cHash = pathcommit.cHash
        cHash = cHash[:7]
        commMsg = "{}\t\t{}\t{}".format( cHash,
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
        'Model'  if  abs(thr00 - thr01) > abs(thr00 - thr10) else  'OV'
        ))
    print("*" * 58" + "\n")
    print("* m1 = {}".format(pathcommit.firstModel))
    print("** m2 = {}".format(pathcommit.secondModel))

def  generateFullConfig(cfg, customCfg):
    tmpJSON = cfg
    # handling input commits
    if  'c'  in  customCfg:
        curCfg = tmpJSON['runConfig']
        interval = customCfg['c'].split("..")
        start, end = interval[0], interval[1]
        curCfg['commitList'] = {'explicitList': [ start, end ] }
    tmpJSON['runConfig'] = curCfg
    # passing base parameters
    CommonTemplate.passParameters([
       'gitPath', 'buildPath', 'appPath'
        ], customCfg, tmpJSON)
    # resolving of API types
    if  isinstance(customCfg['appCmd'], list):
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
```
---
### Configuration and CLI

We designed two interfaces, depending on usercase

API #1 (full command lines, so we can define parameters separately)
```
"template" : {
    "name" : "bm_cc",
    "gitPath":"{gitPath}",
    "buildPath":"{buildPath}",
    "appPath":"{buildPath}",
    "c": "{start}..{end}",
    "appCmd" : ["{appCmd} -m first_model.xml", "{appCmd} -m second_model.xml"]
}
```
API #2 (for unified parameters, only model changes)
```
# Here we use default paths.
"template" : {
    "name" : "bm_cc",
    "c": "{start}..{end}",
    "appCmd" : "./benchmark_app {actualPar} -hint throughput -i input.png",
    "par_1" : "first_model.xml",
    "par_2" : "second_model.xml",
}
```
CLI
```
python3 commit_slider -t bm_cc -c <start>..<end> -appCmd './benchmark_app {actualPar} -hint throughput -i input.png' -par_1 first_model.xml -par_2 second_model.xml
```
### Output
```
python3 commit_slider.py -cfg cfg.json

*****************<Commit slider output>*******************

Table of throughputs:

                     m1      m2

<start_hash>         1000.0  500.0

<end_hash>           1000.0  500.0

**********************************************************

Supposed rootcause is: Model

**********************************************************



*  m1 = ./crossCheckSeparateTemplateBadModel -m good_model.xml

** m2 = ./crossCheckSeparateTemplateBadModel -m bad_model.xml

```

## What we can do the next ?

- Separate Traversal with performance cross-checking.

- Additional step with binary search in the case of ov-caused degradation to find problematic commit.
