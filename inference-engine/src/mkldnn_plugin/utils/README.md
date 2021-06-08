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
    OV_CPU_BLOB_DUMP_DIR=dump_dir binary ...
```
or for shell session (bash example):
```sh
    export OV_CPU_BLOB_DUMP_DIR=dump_dir
    binary ...
```
### Specify dump directory
```sh
    OV_CPU_BLOB_DUMP_DIR=<directory-name> binary ...
```
Default is *mkldnn_dump*
### Specify dump format
```sh
    OV_CPU_BLOB_DUMP_FORMAT=<format> binary ...
```
Options are:
* BIN (default)
* TEXT

### Filter by execution ID
To dump blobs only for node with specified execution IDs:
```sh
    OV_CPU_BLOB_DUMP_NODE_EXEC_ID='<space_separated_list_of_ids>' binary ...
```
Example:
```sh
    OV_CPU_BLOB_DUMP_NODE_EXEC_ID='1 12 45' binary ...
```

### Filter by type
To dump blobs only for node with specified type:
```sh
    OV_CPU_BLOB_DUMP_NODE_TYPE=<type> binary ...
```
Example:
```sh
    OV_CPU_BLOB_DUMP_NODE_TYPE=Convolution binary ...
```

> **NOTE**: see **enum Type** in [mkldnn_node.h](../mkldnn_node.h) for list of the types

### Filter by name
To dump blobs only for node with name matching specified regex:
```sh
    OV_CPU_BLOB_DUMP_NODE_NAME=<regex> binary ...
```
Example:
```sh
    OV_CPU_BLOB_DUMP_NODE_NAME=".+Fused_Add.+" binary ...
```

### Dump all the blobs
```sh
    OV_CPU_BLOB_DUMP_NODE_NAME=".+" binary ...
```

## Graph serialization
The functionality allows to serialize execution graph using environment variable:
```sh
    OV_CPU_EXEC_GRAPH_PATH=<path> binary ...
```

Possible serialization options:
* cout

    Serialize to console output
* \<path\>.xml

    Serialize graph into .xml and .bin files. Can be opened using, for example, *netron* app
* \<path\>.dot

    TBD. Serialize graph into .dot file. Can be inspected using, for example, *graphviz* tools.


