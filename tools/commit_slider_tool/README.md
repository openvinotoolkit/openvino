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

## Setup config

1. Replace `gitPath, buildPath` in *cfg.json* if your target is out of current **Openvino** repo. 
2. Set `appCmd, appPath` regarding target application
3. Set up `specialConfig`:
3.1. `getCommitListCmd` - *git* command, returning commit list *if you don't want to set commit intervals with command args*
3.2. `mode` = `{checkOutput|<to_extend>}` - cryterion of commit comparation
3.3. `traversal` `{firstFailedVersion|firstFixedVersion|<to_extend>}` - traversal rule
3.4. `preprocess` if you need preparation before commit building `<add_details>`
3.5. Other fields depend on mode, for example, `stopPattern`for  `checkOutput` is *RegEx* pattern for application failed output.
## Run commit slider

`python3 commit_slider.py {-c commit1..commit2 | -cfg path_to_config_file}`
`-c` overrides `getCommitListCmd` in *cfg.json*

#### Examples
`python3 commit_slider.py`
`python3 commit_slider.py e29169d..e4cf8ae`
`python3 commit_slider.py e29169d..e4cf8ae my_cfg.json`

## Implementing custom mode
`<todo>`

## Implementing custom traversal rule
`<todo>`