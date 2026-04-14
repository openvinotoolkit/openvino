# IR serialization

IR could be serialized at specified stages using environment variable **OV_CPU_DUMP_IR**:
```sh
    OV_CPU_DUMP_IR=<space_separated_options> binary ...
```

Examples:
```sh
    OV_CPU_DUMP_IR="transformations" binary ...
    OV_CPU_DUMP_IR="transformations=snippets dir=path/dumpDir" binary ...
    OV_CPU_DUMP_IR="transformations=all,-common dir=path/dumpdir formats=svg,xml" binary ...
```

Option names are case insensitive, the following options are supported:
* dir=\<path\>\
Path to dumped IR files. If omitted, it defaults to *intel_cpu_dump*
* formats=<comma_separated_tokens>\
Filter with IR formats to dump. If omitted, it defaults to *xml*\
See [IR format filter](debug_caps_filters.md#ir-format-filter) for more details.
* transformations=<comma_separated_tokens>\
Filter with main transformation stages to serialize graph before and after specified ones.\
See [transformation filter](debug_caps_filters.md#transformation-filter) for more details.

Options are processed from left to right, so last one overwrites previous ones if duplicated.
