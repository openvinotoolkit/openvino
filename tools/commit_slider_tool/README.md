# Commit slider tool

Tool for automatic iteration over commit set with provided operation. For example, binary search with given cryterion (check application output, compare printed blobs, etc.)

## Prerequisites

git >= \*version\*
cmake >= \*version\*
python >= \*version\*

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
4. Set `appCmd, appPath` regarding target application
5. Set up `runConfig`:
5.1. `getCommitListCmd` - *git* command, returning commit list *if you don't want to set commit intervals with command args*
5.2. `mode` = `{checkOutput|bmPerfMode|<to_extend>}` - cryterion of commit comparation
5.3. `traversal` `{firstFailedVersion|firstFixedVersion|<to_extend>}` - traversal rule
5.4. `preprocess` if you need preparation before commit building `<add_details>`
5.5. Other fields depend on mode, for example, `stopPattern` for  `checkOutput` is *RegEx* pattern for application failed output.

## Run commit slider

`python3 commit_slider.py {-c commit1..commit2 | -cfg path_to_config_file}`
`-c` overrides `getCommitListCmd` in *cfg.json*

#### Examples
`python3 commit_slider.py`
`python3 commit_slider.py -c e29169d..e4cf8ae`
`python3 commit_slider.py -c e29169d..e4cf8ae -cfg my_cfg.json`

## Implementing custom mode
`<todo>`

## Implementing custom traversal rule
`<todo>`

## <a name="ccl"></a>Custom command list
`<todo>`