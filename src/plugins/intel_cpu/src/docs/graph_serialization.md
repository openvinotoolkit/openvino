# Graph serialization

Graph serialization is disabled by default and controlled by environment variables.

## Execution graph

Execution graph could be serialized using environment variable **OV_CPU_EXEC_GRAPH_PATH**:
```sh
    OV_CPU_EXEC_GRAPH_PATH=<option> binary ...
```
Possible serialization options:
* cout\
Serialize to console output.
* \<path\>.xml\
Serialize graph into .xml and .bin files. Can be opened using, for example, *netron* app.
* **TBD**: \<path\>.dot\
Serialize graph into .dot file. Can be inspected using, for example, *graphviz* tools.

## Graph transformations

Additionally, execution graph could be serialized at specified stages during creation\
using environment variable **OV_CPU_DUMP_IR**:
```sh
    OV_CPU_DUMP_IR=<space_separated_options> binary ...
```

Examples:
```sh
    OV_CPU_DUMP_IR="transformations" binary ...
    OV_CPU_DUMP_IR="TRANSFORMATIONS=snippets dir=path/dumpDir" binary ...
    OV_CPU_DUMP_IR="transformations=all,-common DIR=path/dumpdir" binary ...
```

Option names are case insensitive, the following options are supported:
* dir=\<path\>\
Path to dumped .xml and .bin files. If omitted, it defaults to *intel_cpu_dump*
* transformations=<comma_separated_tokens>\
Filter with main transformation stages to serialize graph before and after specified ones.\
See [transformation filter](graph_transformation_filter.md) for more details.

If options are duplicated, each one is applied from left to right.
