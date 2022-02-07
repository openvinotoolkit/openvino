# Debug capabilities

## Use the following cmake option to enable debug logs:

`-DENABLE_OPENVINO_DEBUG=ON`

## Use the following cmake option to enable extended debug capabilities:

`-DENABLE_DEBUG_CAPS=ON`

Then, it is possible to

* see an additional debug logs,

* disable snippet transformations by means of the environment variable **OV_ENABLE**:
```sh
    OV_ENABLE=-snippets binary ...
```

* serialize each subgraph before and after lowering by means of the environment variables **OV_DUMP_IR** and **OV_DUMP_IR_DIR**:
```sh
    OV_DUMP_IR=snippets OV_DUMP_IR_DIR=<path> binary ...
```
