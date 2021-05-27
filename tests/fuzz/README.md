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


## Building fuzz tests

1. Build openvino

Build openvino with options `ENABLE_FUZZING` and `ENABLE_SANITIZER` enabled. It
is recommended to use clang compiler.

```bash
(\
mkdir -p build && cd build && \
CC=clang-10 CXX=clang++-10 cmake .. -DENABLE_FUZZING=ON -DENABLE_SANITIZER=ON -DTREAT_WARNING_AS_ERROR=OFF && \
cmake --build . \
)
```

2. Build fuzz tests

Build fuzz tests with options `ENABLE_FUZZING` and `ENABLE_SANITIZER` enabled.
You should use the same compiler as was used for the openvino build.

```bash
(\
mkdir -p tests/fuzz/build && cd tests/fuzz/build && \
CC=clang-10 CXX=clang++-10 cmake .. -DENABLE_FUZZING=ON -DENABLE_SANITIZER=ON -DTREAT_WARNING_AS_ERROR=OFF -DInferenceEngine_DIR=$(pwd)/../../../build && \
cmake --build . \
)
```

## Running fuzz tests

1. Prepare fuzzing corpus

Fuzzing engine needs a set of valid inputs to start fuzzing from. Those files
are called a fuzzing corpus. Use tests/fuzz/scripts/init_corpus.py script to
prepare fuzzing corpus.

```bash
tests/fuzz/scripts/init_corpus.py ./pdpd_layer_models/**/*.pdmodel --join pdiparams && \
mkdir -p import_pdpd-corpus && find ./pdpd_layer_models/ -name "*.fuzz" -exec cp \{\} ./import_pdpd-corpus \;
```

2. Run fuzzing

```bash
OV_FRONTEND_PATH=$(pwd)/lib ./import_pdpd-fuzzer -max_total_time=600 ./import_pdpd-corpus
```
Consider adding those useful command line options:
- `-jobs=$(nproc)` runs multiple fuzzing jobs in parallel.
- `-rss_limit_mb=0` to ignore out-of-memory issues.

## Debugging failures

Fuzzing run halts on the first issue identified, prints issue details to stdout and save data to reproduce the issue as a file in the current folder. To debug the issue pass reproducer as command line argument to fuzz test

```bash
OV_FRONTEND_PATH=$(pwd)/lib ./import_pdpd-fuzzer crash-409b5eeed46a8445b7f7b7a2ce5b60a9ad895e3b
```

It is recommended but not required to use binaries built for fuzzing to debug the issues. A binaries built without `ENABLE_FUZZING` options can also be used to reproduce and debug the issues.
