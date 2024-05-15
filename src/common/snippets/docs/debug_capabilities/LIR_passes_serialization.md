# Snippets LIR passes serialization

LIR(Linear Intermediate Representation) is used as graph reprsentation in control flow pipeline, where dozens of passes are applied to LIR. This is to transfer the graph gradually to the stage that can generate kernel directly via expression instruction emission. When each pass is applied to LIR, there are some expected changes. Developers maybe want to check if the the result is as expected. This capability is introduced to serialize LIRs before and after passes, then developer can check these LIR changes and stages.

To turn on snippets LIR passses serialization feature, the following environment variable should be used:
```sh
    OV_SNIPPETS_DUMP_LIR=<space_separated_options> binary ...
```

Examples:
```sh
    OV_SNIPPETS_DUMP_LIR="passes=all dir=path/dumpdir formats=all" binary ...
    OV_SNIPPETS_DUMP_LIR="passes=Insert dir=path/dumpdir formats=control_flow" binary ...
    OV_SNIPPETS_DUMP_LIR="passes=Insert formats=data_flow" binary ...
```

Option names are case insensitive, the following options are supported:
 - `passes` : Dump LIR around the pass if the set value is included in pass name. Key word 'all' means to dump LIR around every pass. If not set, will not dump.
 - `dir` : Path to dumped LIR files. If omitted, it defaults to intel_snippets_LIR_dump.
 - `formats` : Support value of control_flow, data_flow or all. If omitted, it defaults to control_flow.