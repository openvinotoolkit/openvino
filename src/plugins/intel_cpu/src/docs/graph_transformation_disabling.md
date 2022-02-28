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
See [transformation filter](debug_caps_filters.md#transformation-filter) for more details.

Options are processed from left to right, so last one overwrites previous ones if duplicated.
