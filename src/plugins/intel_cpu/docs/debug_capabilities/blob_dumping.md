# Blob dumping
Blob dumping is controlled by environment variables (filters).

The variables define conditions of the node which input and output blobs
should be dumped for.

> **NOTE**: Nothing is dumped by default

> **NOTE**: All specified filters should be matched in order blobs to be dumped

Environment variables can be set per execution, for example:
```sh
    OV_CPU_BLOB_DUMP_DIR=dump_dir OV_CPU_BLOB_DUMP_FORMAT=TEXT OV_CPU_BLOB_DUMP_NODE_PORTS=OUT binary ...
```
or for shell session (bash example):
```sh
    export OV_CPU_BLOB_DUMP_DIR=dump_dir
    export OV_CPU_BLOB_DUMP_FORMAT=TEXT
    export OV_CPU_BLOB_DUMP_NODE_PORTS=OUT
    binary ...
```
## Specify dump directory
```sh
    OV_CPU_BLOB_DUMP_DIR=<directory-name> binary ...
```
Default is *cpu_dump*
## Specify dump format
```sh
    OV_CPU_BLOB_DUMP_FORMAT=<format> binary ...
```
Options are:
* BIN
* TEXT (default)

## Filter input / output blobs
To dump only input / output blobs:
```sh
    OV_CPU_BLOB_DUMP_NODE_PORTS='<ports_kind>' binary ...
```
Example:
```sh
    OV_CPU_BLOB_DUMP_NODE_PORTS=OUT binary ...
```
Options are:
* IN
* OUT
* ALL

## Filter by execution ID
To dump blobs only for nodes with specified execution IDs:
```sh
    OV_CPU_BLOB_DUMP_NODE_EXEC_ID='<space_separated_list_of_ids>' binary ...
```
Example:
```sh
    OV_CPU_BLOB_DUMP_NODE_EXEC_ID='1 12 45' binary ...
```

## Filter by type
To dump blobs only for nodes with specified types:
```sh
    OV_CPU_BLOB_DUMP_NODE_TYPE=<space_separated_list_of_types> binary ...
```
Example:
```sh
    OV_CPU_BLOB_DUMP_NODE_TYPE='Convolution Reorder' binary ...
```

> **NOTE**: see **enum Type** in [node.h](../../src/cpu_types.h) for list of the types

## Filter by name
To dump blobs only for nodes with name matching specified regex:
```sh
    OV_CPU_BLOB_DUMP_NODE_NAME=<regex> binary ...
```
Example:
```sh
    OV_CPU_BLOB_DUMP_NODE_NAME=".+Fused_Add.+" binary ...
```

## Dump all the blobs
```sh
    OV_CPU_BLOB_DUMP_NODE_NAME="*" binary ...
```
    or
```sh
    OV_CPU_BLOB_DUMP_NODE_NAME=".+" binary ...
```
    or
```sh
    OV_CPU_BLOB_DUMP_NODE_PORTS=ALL binary ...
```
