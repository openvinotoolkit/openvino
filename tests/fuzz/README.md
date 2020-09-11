# Fuzzing Test Suite

This test suite contains [fuzzing](https://en.wikipedia.org/wiki/Fuzzing) tests for [libFuzzer](https://llvm.org/docs/LibFuzzer.html) fuzzing engine.

## Getting Started

Each fuzzing test is an executable. It can run fuzzing to search for new
failures and save reproducer in a file. You can later run a fuzzing test with a
reproducer to debug a failure found.

## Pre-requisites

There are no special pre-requisites to reproduce and debug failures.

To run fuzzing you will need [LLVM](https://apt.llvm.org/) components:
- Clang and co.
- libFuzzer
- lld (linker)
- libc++

## Reproducing Failure Found by Fuzzing

1. Build `fuzz` test target:
```bash
cmake -DENABLE_TESTS=ON .. && ninja fuzz
```

2. Run fuzzing test passing a failure reproducer as a command-line argument:
``` bash
./read_network-fuzzer crash-reproducer
```
