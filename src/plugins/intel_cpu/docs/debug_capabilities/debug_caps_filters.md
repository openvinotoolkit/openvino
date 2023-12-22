# Filters

Filters described below have the following common format:
```sh
    filter_name=<comma_separated_tokens>
```
Tokens are processed from left to right and each one includes or excludes corresponding value.\
For exclusion token is just prepended by minus: *-token*\
All tokens are case insensitive and no tokens is treated as *all*\
So filters below are equal:
* filter_name
* filter_name=all
* filter_name=-all,ALL

## IR format filter

IR format filter is used to specify output IR formats, e.g. for [serialization](graph_serialization.md#graph-transformations).
```sh
    formats=<comma_separated_tokens>
```

The following tokens are supported:
* all\
equals to <xml,dot,svg>
* xml (default)\
IR in .xml file. Can be opened using, for example, *netron* app. (For now the option is Linux only)
* xmlbin\
IR in .xml and .bin files. Can be opened using, for example, *netron* app. Also can be read using OV API
* dot\
IR in .dot file (.svg.dot file if svg is also specified). Can be inspected using, for example, *graphviz* tools.
* svg\
IR in .svg file. Requires *dot* tool to be installed on the host, not supported on Windows.\
Generation is based on dot representation, so IR is additionally dumped to .svg.dot file.

## Transformation filter

Transformation filter is used to specify main graph transformation stages for different purposes,
e.g. for [disabling](feature_disabling.md#graph-transformation) or [serialization](graph_serialization.md#graph-transformations).
```sh
    transformations=<comma_separated_tokens>
```

The following tokens are supported:
* all (default)\
equals to <preLpt,lpt,postLpt,snippets,specific>
* common \
equals to <preLpt,postLpt>
* preLpt
* lpt
* postLpt
* snippets
* specific
