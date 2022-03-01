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

* serialize each subgraph before and after lowering by means of the environment variable **OV_SNIPPETS_DUMP_IR**:
```sh
    OV_SNIPPETS_DUMP_IR=<space_separated_options> binary ...
```

### Subgraph serialization

Examples:
```sh
    OV_SNIPPETS_DUMP_IR="dir=path/dumpDir" binary ...
    OV_SNIPPETS_DUMP_IR="formats=svg,xml" binary ...
    OV_SNIPPETS_DUMP_IR="DIR=path/dumpdir formats" binary ...
```

Option names are case insensitive, the following options are supported:
* dir=\<path\>\
Path to dumped IR files. If omitted, it defaults to *snippets_dump*
* formats=<comma_separated_tokens>\
Filter with IR formats to dump. If omitted, it defaults to *.dot*\
Tokens are processed from left to right and each one includes or excludes corresponding format.\
For exclusion token is just prepended by minus: *-token*\
All tokens are case insensitive and no tokens is treated as *all*\
The following tokens are supported:
    * all\
equals to <xml,dot,svg>
    * xml\
IR in .xml and .bin files. Can be opened using, for example, *netron* app.
    * dot\
IR in .dot file (.svg.dot file if svg is also specified). Can be inspected using, for example, *graphviz* tools.
    * svg\
IR in .svg file. Requires *dot* tool to be installed on the host, not supported on Windows.\
Generation is based on dot representation, so IR is additionally dumped to .svg.dot file.

Options are processed from left to right, so last one overwrites previous ones if duplicated.
