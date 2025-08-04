# Broken Compilation

`broken_compilation` — template for searching of eirst broken commit (in a sense of failed compilation stage).

- Cmake-way of building (without wheels)
- Looking for compilation (not linking) errors

---

## How to run
Example 1 (CLI)
```
python3 commit_slider.py -t broken_compilation -c <start_commit>..<end_commit> -gitPath <pathToGitRepository> -buildPath <pathToBuildDirectory>
```
Example 2 (Configuration, default paths)
```
{ "template" : {
    "name" : "broken_compilation",
    "c" : "<start_commit>..<end_commit>"
}
```
### Independent variables:
```
- verbosity: -v {true|false}
```

### Expected output:
```
*****************<Commit slider output>*******************
* Commit with broken compilation found:                  *
* Break commit: <comit_hash>, state: CommitState.BREAK   *
**********************************************************
```