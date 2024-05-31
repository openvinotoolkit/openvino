# how to enable tensor parallel in openvino?

`ENABLE_TP` is to control the tensor parallel mode, you can set as below:

- `export ENABLE_TP=1` : split src and wgt, element-add dst
- `export ENABLE_TP=2` : split src and wgt, concat dst in horizontal direction
- `export ENABLE_TP=3` : split src(batch size > 1 is required.), concat dst in vertical direction

# how to turn on profiler tool?

`CPU_PROFILE` is to control `profile.json` dumping, which can be open in Chrome or Edge browser.

- `export CPU_PROFILE=1` :  turn on the tracing file dump option.
- `unset CPU_PROFILE` :  turn off dump option.