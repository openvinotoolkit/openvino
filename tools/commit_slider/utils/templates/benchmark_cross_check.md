
# Performance cross - check

`bm_cc` — template for checking if degradation caused ov-version or  model.

---
  

## How to run

Example 1
```
python3 commit_slider -t bm_cc -c <start>..<end> -appCmd './benchmark_app {actualPar} -hint throughput -i input.png' -par_1 first_model.xml -par_2 second_model.xml -gitPath <pathToGitRepository> -buildPath <pathToBuildDirectory>
```
Example 2 (with configuration)
```
python3 commit_slider -cfg custom_cfg.json
```
where
```
cfg = { "template" : { 
    "name" : "bm_cc",
    "c" : "<start>..<end>",
    "appCmd" : [
        "./benchmark_app -m model_1 -hint throughput -i input_1.png",
        "./benchmark_app -m model_2 -hint throughput -i input_2.png"
    ]
}}
```
### Independent variables:

```
- verbosity: -v {true|false}
```
### Expected output:

```
*****************<Commit slider output>*******************
Table of throughputs:
             m1               m2
<start_hash> <throughput_00>  <throughput_01>
<end_hash>   <throughput_10>  <throughput_11>
**********************************************************
Supposed rootcause is: <Model | OV>
**********************************************************

*  m1 = ./benchmark_app -m first_model.xml
** m2 = ./benchmark_app -m second_model.xml
```