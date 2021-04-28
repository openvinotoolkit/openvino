# Debug capabilities
Use the following cmake option to enable debug capabilities:

`-DENABLE_CPU_DEBUG_CAPS=ON`

## Blob dumping
Blob dumping is controlled by environment variables (filters).

The variables define conditions of the node which input, output and internal blobs
should be dumped for.

> **NOTE**: Nothing is dumped by default

> **NOTE**: All specified filters should be matched in order blobs to be dumped

Environment variables can be set per execution, for example:
```sh
    OV_CPU_BLOB_DUMP_DIR=dump_dir benchmark_app ...
```
or for shell session (bash example):
```sh
    export OV_CPU_BLOB_DUMP_DIR=dump_dir
    benchmark_app ...
```
### Specify dump directory
```sh
    OV_CPU_BLOB_DUMP_DIR=<directory-name> benchmark_app ...
```
Default is *mkldnn_dump*
### Specify dump format
```sh
    OV_CPU_BLOB_DUMP_FORMAT=<format> benchmark_app ...
```
Options are:
* BIN (default)
* TEXT

### Filter by execution ID
To dump blobs only for node with specified execution IDs:
```sh
    OV_CPU_BLOB_DUMP_NODE_EXEC_ID='<space_separated_list_of_ids>' benchmark_app ...
```
Example:
```sh
    OV_CPU_BLOB_DUMP_NODE_EXEC_ID='1 12 45' benchmark_app ...
```

### Filter by type
To dump blobs only for node with specified type:
```sh
    OV_CPU_BLOB_DUMP_NODE_TYPE=<type> benchmark_app ...
```
Example:
```sh
    OV_CPU_BLOB_DUMP_NODE_TYPE=Convolution benchmark_app ...
```

> **NOTE**: see **enum Type** in [mkldnn_node.h](https://github.com/openvinotoolkit/openvino/blob/master/inference-engine/src/mkldnn_plugin/mkldnn_node.h) for list of the types

### Filter by name
To dump blobs only for node with name matching specified regex:
```sh
    OV_CPU_BLOB_DUMP_NODE_NAME=<regex> benchmark_app ...
```
Example:
```sh
    OV_CPU_BLOB_DUMP_NODE_NAME=".+Fused_Add.+" benchmark_app ...
```

### Dump all the blobs
```sh
    OV_CPU_BLOB_DUMP_NODE_NAME=".+" benchmark_app ...
```
