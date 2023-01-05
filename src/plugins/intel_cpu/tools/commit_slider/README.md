# Commit slider tool

Tool for automatic iteration over commit set with provided operation. For example, binary search with given cryterion (check application output, compare printed blobs, etc.)

## Prerequisites

git >= *2.0*
cmake >= OpenVino minimum required version ([CMakeLists.txt](../../../../CMakeLists.txt))
python >= *3.6*
ccache >= *3.0*

## Preparing

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
2. Define `commandList`. Adjust *commandList* if you need more specific way to build target app. More details in [Custom command list](#ccl).
3. Replace `gitPath, buildPath` if your target is out of current **Openvino** repo. 
4. Set `appCmd, appPath` (mandatory) regarding target application
5. Set up `runConfig` (mandatory):
5.1. `getCommitListCmd` - *git* command, returning commit list *if you don't want to set commit intervals with command args*
5.2. `mode` = `{checkOutput|bmPerfMode|compareBlobs|<to_extend>}` - cryterion of commit comparation
5.3. `traversal` `{firstFailedVersion|firstFixedVersion|allBreaks|<to_extend>}` - traversal rule
5.4. `preprocess` if you need preparation before commit building `<add_details>`
5.5. Other fields depend on mode, for example, `stopPattern` for  `checkOutput` is *RegEx* pattern for application failed output.
6. Setup environment variables via *envVars* field in a format:
`[{"name" : "key1", "val" : "val1"}, {"name" : "key2", "val" : "val2"}]`

## Run commit slider

`python3 commit_slider.py {-c commit1..commit2 | -cfg path_to_config_file}`
`-c` overrides `getCommitListCmd` in *cfg.json*

#### Examples

##### Command line
`python3 commit_slider.py`
`python3 commit_slider.py -c e29169d..e4cf8ae`
`python3 commit_slider.py -c e29169d..e4cf8ae -cfg my_cfg.json`

##### Custom configuration
###### Performance task
*custom_cfg.json*
```{
    "appCmd" : "./benchmark_app <params>",
    "makeCmd" : "cmake <cmake_params> ..",
    "runConfig" : {
        "commitList" : {
            "getCommitListCmd" : "git log c1..c2 --boundary --pretty=\"%h\""
        },
        "mode" : "bmPerf",
        "traversal" : "firstFailedVersion",
        "perfAppropriateDeviation" : 0.05
    }
}
```
###### Comparation of blobs
*custom_cfg.json*
```
{
    "appCmd" : "./benchmark_app <params>",
    "makeCmd" : "cmake <cmake_params> ..",
    "envVars" : [
        {"name" : "OV_CPU_BLOB_DUMP", "val" : "Output"},
        {"name" : "OV_CPU_BLOB_DUMP_FORMAT", "val" : "TEXT"},
        {"name" : "OV_CPU_BLOB_DUMP_DIR", "val" : "<path_to_blobs>"}
    ],
    "runConfig" : {
        "mode" : "compareBlobs",
        "traversal" : "allBreaks",
        "outputFileNamePattern" : "^sink_mask_0.ieb$",
        "outputDirectory" : "<path_to_blobs>",
        "limit" : 0.02
    }
}
```

###### Check output
*custom_cfg.json*
```
{
    "appCmd" : "<application>",
    "makeCmd" : "cmake <cmake_params> ..",
    "runConfig" :
        "mode" : "checkOutput",
        "traversal" : "firstFailedVersion",
        "stopPattern" : "(.)*fail(.)*"
    }
}
```

## Implementing custom mode
`<todo>`

## Implementing custom traversal rule
`<todo>`

## <a name="ccl"></a>Custom command list
`<todo>`