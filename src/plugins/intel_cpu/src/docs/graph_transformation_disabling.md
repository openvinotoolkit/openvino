# Graph transformation disabling

Graph transformation disabling is controlled by environment variable **OV_CPU_DISABLE**:
```sh
    OV_CPU_DISABLE=<space_separated_options> binary ...
```

Examples:
```sh
    OV_CPU_DISABLE="transformations" binary ...
    OV_CPU_DISABLE="TRANSFORMATIONS=lpt" binary ...
    OV_CPU_DISABLE="transformations=all,-common" binary ...
```

Option names are case insensitive, the following options are supported:
* transformations=<comma_separated_tokens>\
Filter with main transformation stages to disable specified ones.\
See [transformation filter](graph_transformation_filter.md) for more details.

If options are duplicated, each one is applied from left to right.
