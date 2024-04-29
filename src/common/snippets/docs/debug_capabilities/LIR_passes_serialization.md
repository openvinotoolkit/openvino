# Snippets LIR passes serialization

LIR(Linear Intermediate Representation) is used as the graph format in control flow pipeline, where dozens of passes are applied to LIR. This is to transfer the graph gradually to the stage that can generate kernel directly via expression emit. When each pass applied to LIR, there are some changes to the LIR. Developers maybe want check if the the result is as expected. This capability is introduced to serialize LIRs after every pass, then developer can check all these LIR stages.

To turn on snippets LIR passses serialization feature, the following environment variable should be used to set the folder that the LIRs is dumped to:
```sh
    OV_CPU_SNIPPETS_LIR_PATH=<folder_path> binary ...
```

Examples:
```sh
    OV_CPU_SNIPPETS_LIR_PATH="/home/debug/LIRs" binary ...
    OV_CPU_SNIPPETS_LIR_PATH="/home/debug/LIRs/" binary ...
```