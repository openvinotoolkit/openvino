# Transformation filter

Transformation filter is used to specify main graph transformation stages for different purposes,
e.g. for [disabling](graph_transformation_disabling.md) or [serialization](graph_serialization.md).
```sh
    transformations=<comma_separated_tokens>
```

Tokens are processed from left to right and each one includes or excludes corresponding graph transformation stages.\
For exclusion token is just prepended by minus: -token. The following tokens are supported:
* all\
equals to <common,specific>
* common\
equals to <preLpt,lpt,postLpt,snippets>
* preLpt
* lpt
* postLpt
* snippets
* specific

All tokens are case insensitive and no tokens is treated as all, so filters below are equal:
* transformations
* transformations=all
* transformations=common,specific
* transformations=-all,ALL
* transformations=-all,common,specific
* transformations=-ALL,cOmMoN,SpEcIfIc
