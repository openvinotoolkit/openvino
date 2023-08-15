# Feature disabling

Common way to disable something in CPU plugin is implied by means of environment variable **OV_CPU_DISABLE**:
```sh
    OV_CPU_DISABLE=<space_separated_options> binary ...
```
Option names are case insensitive and processed from left to right,\
so last one overwrites previous ones if duplicated.

Examples:
```sh
    OV_CPU_DISABLE="transformations" binary ...
    OV_CPU_DISABLE="transformations=lpt" binary ...
    OV_CPU_DISABLE="transformations=all,-common" binary ...
```

By means of corresponding options **OV_CPU_DISABLE** controls disabling of the following features:

## Graph transformations

Graph transformation disabling is controlled by the following option inside **OV_CPU_DISABLE**:
```sh
transformations=<comma_separated_tokens>
```
Filter with main transformation stages to disable specified ones.\
See [transformation filter](debug_caps_filters.md#transformation-filter) for more details.
