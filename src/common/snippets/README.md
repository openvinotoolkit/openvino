# Debug capabilities

## Use the following cmake option to enable debug logs:

`-DENABLE_OPENVINO_DEBUG=ON`

## Use the following cmake option to enable extended debug capabilities:

`-DENABLE_DEBUG_CAPS=ON`

Then, it is possible to

* see an additional debug logs,

* disable snippet transformations by means of the environment variable **OV_SNIPPETS_DISABLE**:
```sh
    OV_SNIPPETS_DISABLE=ON binary ...
```

* serialize each subgraph before and after lowering by means of the environment variable **OV_SNIPPETS_DUMP_IR_DIR**:
```sh
    OV_SNIPPETS_DUMP_IR_DIR=<path> binary ...
```
