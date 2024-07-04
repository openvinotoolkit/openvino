# Snippets LIR passes serialization

LIR(Linear Intermediate Representation) is used as graph reprsentation in control flow pipeline, where dozens of passes are applied to LIR. This is to transfer the graph gradually to the stage that can generate kernel directly via expression instruction emission. When each pass is applied to LIR, there are some expected changes. Developers maybe want to check if the result is as expected. This capability is introduced to serialize LIRs before and after passes, then developer can check these LIR changes and status.

To turn on snippets LIR passses serialization feature, the following environment variable should be used:
```sh
    OV_SNIPPETS_DUMP_LIR=<space_separated_options> binary ...
```

Examples:
```sh
    OV_SNIPPETS_DUMP_LIR="passes=ExtractLoopInvariants dir=path/dumpdir formats=all" binary ...
    OV_SNIPPETS_DUMP_LIR="passes=all dir=path/dumpdir formats=control_flow" binary ...
    OV_SNIPPETS_DUMP_LIR="passes=FuseLoops,InsertLoops,InsertLoadStore formats=data_flow" binary ...
```

Option names are case insensitive, the following options are supported:
 - `passes` : Dump LIR around the passes if passes name are specified. It support mutiple passes with comma separated and pass name is case insensitive. Key word 'all' means to dump LIR around every pass. This option is a must have, should not be omitted.
 - `dir` : Path to dumped LIR files. If omitted, it defaults to snippets_LIR_dump.
 - `formats` : Support values are control_flow, data_flow and all. If omitted, it defaults to control_flow.