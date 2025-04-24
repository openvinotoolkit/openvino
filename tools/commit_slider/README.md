# Commit slider tool

Tool for automatic iteration over commit set with provided operation. For example, binary search with given cryterion (check application output, compare printed blobs, etc.)

## Prerequisites

git >= *2.0*
cmake >= OpenVino minimum required version ([CMakeLists.txt](../CMakeLists.txt))
python >= *3.6*
ccache >= *3.0*

## Preparing (only for *Linux*)

 1. Install **CCache**:
`sudo apt install -y ccache`
`sudo /usr/sbin/update-ccache-symlinks`
`echo 'export PATH="/usr/lib/ccache:$PATH"' | tee -a ~/.bashrc`
`source ~/.bashrc && echo $PATH`
2. Check if **Ccache** installed via `which g++ gcc`
3. Run `sudo sh setup.sh`

## Setup custom config

*custom_cfg.json* may override every field in general *util/cfg.json*. Here are the most necessary.

1. Define `makeCmd` - build command, which you need for your application.
2. Define `commandList`. Adjust *commandList* if you need more specific way to build target app. In a case of *Win OS* it's reasonable to override `commandList` with specific make command, like `cmake --build . --config Release` after `{makeCmd}`. More details in [Custom command list](#ccl).
3. Replace `gitPath, buildPath` if your target is out of current **Openvino** repo. 
4. Set `appCmd, appPath` (mandatory) regarding target application
5. Set up `runConfig` (mandatory):
5.1. `getCommitListCmd` - *git* command, returning commit list *if you don't want to set commit intervals with command args* or `explicitList` if you want to set up commits manually.
Examples:
```
"commitList" : {
    "getCommitListCmd" : "git log start_hash..end_hash --boundary --pretty=\"%h\""
}
```
```
"commitList" : {
    "explicitList" : ["hash_1", "hash_2", ... , "hash_N"]
}
```
5.2. `mode` = `{checkOutput|bmPerfMode|compareBlobs|<to_extend>}` - cryterion of commit comparation
5.3. `traversal` `{firstFailedVersion|firstFixedVersion|allBreaks|bruteForce|<to_extend>}` - traversal rule
5.4. `preprocess` if you need preparation before commit building.
5.5. Other fields depend on mode, for example, `stopPattern` for  `checkOutput` is *RegEx* pattern for application failed output.
6. Setup environment variables via *envVars* field in a format:
`[{"name" : "key1", "val" : "val1"}, {"name" : "key2", "val" : "val2"}]`
7. setup [Prebuilded apps config](#pba).

## Run commit slider

`python3 commit_slider.py {-c commit1..commit2 | -cfg path_to_config_file}`
`-c` overrides `getCommitListCmd` in *cfg.json*

## Output

In common case, the output looks like
```
    <build details>
    Break commit: "hash_1", <details>
    Break commit: "hash_2", <details>
    <...>
    Break commit: "hash_N", <details>
```
For every *x* *hash_x* means hash of commit, caused break, i.e. previous commit is "good" in a sense of current Mode, and *hash_x* is "bad". `<details>` determined by Mode. Common log and log for every commit are located in *log/*. If `printCSV` flag is enabled, *log/* contains also *report.csv*.

## Examples

### Command line
`python3 commit_slider.py`
`python3 commit_slider.py -c e29169d..e4cf8ae`
`python3 commit_slider.py -c e29169d..e4cf8ae -cfg my_cfg.json`


### Predefined configurations
There are several predefined configurations in *utils/cfg_samples* folder, which may me copied to *custom_cfg.json*. In every example `<start_commit>..<end_commit>` means interval to be analized. All examples illusrate the simpliest binary search.

###### Performance task
Searching of performance degradation of *benchmark_app*.
*bm_perf.json*
`<model_path>` - path to model for benchmarking, `perfAppropriateDeviation` may be changed to make acceptance condition more strict or soft. 

###### Comparation of blobs
Checking of accuracy degradation via blob comparation.
*blob.cmp.json*
`<model_path>` - path to model, `<blob_dir>` - directory for printing blobs, `<out_blob_name_tail>` - pattern of blob name, not including node id, for example *Output_Reshape_2-0_in0*, `limit` of linewise difference may be changed or zeroed for bitwise comparation, `OV_CPU_BLOB_DUMP_NODE_TYPE` corresponds required node type, other dumping parameters may be also used.

###### Check output
Searching of failing of *benchmark_app*.
*custom_cfg.json*
`<model_path>` - path to model, `<bm_error_message>` - typical error message or part of it, e.g. *fail*.

###### Integration with e2e
Checking of accuracy degradation by *e2e*. `<e2e_path>` - path to e2e directory, `<e2e_args>` - parameters for e2e, `<ov_path>` - absolute path to *ov*, `<e2e_error_message>` - e2e error message.


## Possible troubles
In the case of unexpected failing, you can check */tmp/commit_slider/log/*
###### Insufficient build commandset
If some commit cannot be builded, you can extend command set in custom command list. The example of custom commandlist is below:
```
"commandList" : [
    {"cmd" : "git rm --cached -r .", "path" : "{gitPath}", "tag" : "clean"},
    {"cmd" : "git reset --hard", "path" : "{gitPath}", "tag" : "clean"},
    {"cmd" : "git rm .gitattributes", "path" : "{gitPath}", "tag" : "clean"},
    {"cmd" : "git reset .", "path" : "{gitPath}", "tag" : "clean"},
    {"cmd" : "git checkout -- .", "path" : "{gitPath}", "tag" : "clean"},
    {"cmd" : "git rm --cached -r .", "path" : "{gitPath}", "tag" : "clean"},
    {"cmd" : "git reset --hard", "path" : "{gitPath}", "tag" : "clean"},
    {"cmd" : "git rm .gitattributes", "path" : "{gitPath}", "tag" : "clean"},
    {"cmd" : "git reset .", "path" : "{gitPath}", "tag" : "clean"},
    {"cmd" : "git checkout -- .", "path" : "{gitPath}"},
    {"cmd" : "git clean -fxd", "path" : "{gitPath}", "tag" : "clean"},
    {"cmd" : "mkdir -p build", "path" : "{gitPath}"},
    {"cmd" : "git checkout {commit}", "catchMsg" : "error", "path" : "{gitPath}"},
    {"cmd" : "git submodule init", "path" : "{gitPath}"},
    {"cmd" : "git submodule update --recursive", "path" : "{buildPath}"},
    {"cmd" : "{makeCmd}", "catchMsg" : "CMake Error", "path" : "{buildPath}"},
    {"cmd" : "make --jobs=4", "path" : "{buildPath}"},
    {"cmd" : "git checkout -- .", "path" : "{gitPath}"}
]
```
More details in [Custom command list](#ccl).

###### Application failed with another or output wasn't parsed correctly
Sometimes, target bug is covered by another unexpected bug. In this case, it's reasonable to extend error pattern, like {err_msg_1|err_msg_2} or look for new problem with separate run.

## Implementing custom mode
1. override `checkIfBordersDiffer(i1, i2, list)` to define, if given commits differs in terms of given criterion. 
2. override `createCash(), getCommitIfCashed(commit), getCommitCash(commit, valueToCache)` if predefined cashing via json map is insufficient for current task.
3. `checkCfg()` - checking if customer provided all necessary parameters.
4. `setOutputInfo(commit), getResult()` for setting and interpretation of parameters of founded commit.


## Implementing custom traversal rule
To implement new `Traversal`, override `bypass(i1, i2, list, cfg, commitPath)` method, using `checkIfBordersDiffer(i1, i2, list)` from Mode to decide, whether desired commit is lying inside given interval.

## Implementing custom preprocess
1. create *utils/preprocess/your_custom_pp.py* file.
2. define `def your_custom_pp(cfg): <...>` function with implementation of subprocess.
3. add `"preprocess" : { "name" : your_custom_pp, <other parameters>` to `runConfig` and `{"tag" : "preprocess"}` to build command list.

## <a name="pba"></a>Using of pre-builded apps

'cachedPathConfig' option helps to speed up searching or to solve environment/building problems. There are two schemas: 'optional', which tunes bypass, if provided paths permit it and build commit in the other case. The aim is to increase performance and solve long-interval rebuilding issues (is to be implemented). 'Mandatory' supposes using only provided paths. Bypass is not impacted. Absent commits are marked as ignored (supposed to contain insignificant changes). Mostly intended for using with complex environment.

1. add `cachedPathConfig` field to config
2. set up `enable` and `scheme` fields
3. define `cashMap`.
4. `passCmdList` flag is true if commandList supposed to be ignored (no build is necessary)
5. `changeAppPath` flag means, that cashed path substitutes `appPath`

Example:
```
"cachedPathConfig": {
    "enable" : true,
    "scheme" : "mandatory",
    "cashMap" : {
        "hash_1_": "app/path/hash_1_",
        "hash_2_": "app/path/hash_2_",
        ..............................
        "hash_N_": "app/path/hash_N_"
    }
}
```

## <a name="ccl"></a>Custom command list
The structure of build command is
```
"commandList" : [
        {
        "cmd" : "git rm --cached -r .",
        "path" : "directory where command is to be runned",
        "tag" : "is used for managing of command flow (clean, preprocessing)",
        "catchMsg" : "RegEx, like ‘(.)*(error|failed|wrong executor)(.)*’"
        },
    <...>
    ]
```
*cmd* - command to run, e.g. `git rm --cached -r .`, *path* - command directory, commonly git root or build directory, *tag* - necessary to check, if command should be executed with some special conditions, commonly `preprocess` or `clean`, *catchMsg* - string to check output, necessary because of unreliability of exceptions handling in python subprocess API.
