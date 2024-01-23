# Snippets segfault detector

Subgraph in snippets is decomposed to many simple operations. These operations are converted to corresponding emitters to generate execution instruction. If a segfault happens during a subgraph execution, it often requires a significant effort to debug and investigate the problem. This capability is introduced to identify the faulty emitter among the large kernel, and to print some useful emitter information.

To turn on snippets segfault detector, the following environment variable should be used:
```sh
    OV_CPU_SNIPPETS_SEGFAULT_DETECTOR=<level> binary ...
```

Currently snippets segfault detector has only one level, any digit can be used for activation.
Currently snippets segfault detector is only effective for x86 or x86-64 CPU backend.