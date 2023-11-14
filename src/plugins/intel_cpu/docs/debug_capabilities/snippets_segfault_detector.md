# Snippets segfault detector

Subgraph in snippets is decomposed to many simple operations. The operations are converted to corresponding emitters to generate execution instruction. If there is a segfault happened in subgraph execution, it would take big effort to debug and investigate. This capability is introduced to identify the blamed emitter among the large kernel, and to cout useful blamed emitter information.

To turn on snippets segfault detector, the following environment variable should be used:
```sh
    OV_CPU_SNIPPETS_SEGFAULT_DETECTOR=<level> binary ...
```

Currently snippets segfault detector has only one level, any digit can be used for activation.