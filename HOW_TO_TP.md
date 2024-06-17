# how to enable tensor parallel in openvino?

`ENABLE_TP` is to control the tensor parallel mode, you can set as below:

- `export ENABLE_TP=ON` : split src and wgt, concat dst in horizontal direction
- `unset ENABLE_TP` :  turn off tensor parallel.

# how to turn on profiler tool?

`CPU_PROFILE` is to control `profile.json` dumping, which can be open in Chrome or Edge browser.

- `export CPU_PROFILE=1` :  turn on the tracing file dump option.
- `unset CPU_PROFILE` :  turn off dump option.
