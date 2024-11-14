# Snippets LIR passes serialization

LIR (Linear Intermediate Representation) is used as graph representation in the control flow pipeline, where dozens of passes are applied to LIR.
This is to transform the graph gradually to the lowered state that can be used for code emission, i.e. every expression can be mapped to a sequence of assembly instructions.
When each pass is applied to LIR, there are some expected changes.
Developers may want to check if the result of a certain transformation pass is expected.
This capability is introduced to serialize LIRs before and after the passes, then a developer can check these LIR changes introduced by the each pass, to see if expected expression inserted, expression order is correct, or expression loop id is correct and so on.

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
 - `passes` : Dump LIR around the passes if passes name are specified.
 It support multiple comma separated pass names. The names are case insensitive.
 Key word 'all' means to dump LIR around every pass.
 This option is a must have, should not be omitted.
 - `dir` : Path to dumped LIR files.
 If omitted, it defaults to snippets_LIR_dump.
 If specified path doesn't exist, the directory will be created automatically.
 - `formats` : Support values are control_flow, data_flow and all.
 If omitted, it defaults to control_flow.