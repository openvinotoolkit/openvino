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
CC=clang CXX=clang++ cmake .. -DENABLE_FUZZING=ON -DENABLE_SANITIZER=ON && \
cmake --build . \
)
```

2. Build fuzz tests

Build fuzz tests with options `ENABLE_FUZZING` and `ENABLE_SANITIZER` enabled.
You should use the same compiler as was used for the openvino build.

```bash
(\
mkdir -p tests/fuzz/build && cd tests/fuzz/build && \
CC=clang CXX=clang++ cmake .. -DENABLE_FUZZING=ON -DENABLE_SANITIZER=ON -DOpenVINO_DIR=$(pwd)/../../../build && \
cmake --build . \
)
```

## Running fuzz tests

1. Prepare fuzzing corpus

Fuzzing engine needs a set of valid inputs to start fuzzing from. Those files
are called a fuzzing corpus. Place valid inputs for the fuzzing test into
directory.

Intel employees can get the corpus as described here
https://wiki.ith.intel.com/x/2N42bg.

2. Run fuzzing

```bash
# LD_PRELOAD is required when OpenVINO build as shared library, the ASAN library has to be pre-loaded.
[LD_PRELOAD=path-to-asan-lib] ./read_network-fuzzer -max_total_time=600 ./read_network-corpus
```

Consider adding those useful command line options:
- `-jobs=$(nproc)` runs multiple fuzzing jobs in parallel. Note: configuring code coverage profiling with environment variable `LLVM_PROFILE_FILE=deafult-%p.profraw` is required.
- `-rss_limit_mb=0` to ignore out-of-memory issues.

## Analyzing fuzzing quality

### Explore code coverage

To build coverage report after fuzz test execution run:

```
llvm-profdata merge -sparse *.profraw -o default.profdata && \
llvm-cov show ./read_network-fuzzer -object=lib/libopenvino.so -instr-profile=default.profdata -format=html -output-dir=read_network-coverage
```

## Reproducing findings

Fuzzing run halts on the first issue identified, prints issue details to stdout and save data to reproduce the issue as a file in the current folder. To debug the issue pass reproducer as command line argument to fuzz test

```bash
./read_network-fuzzer crash-409b5eeed46a8445b7f7b7a2ce5b60a9ad895e3b
```

It is recommended but not required to use binaries built for fuzzing to debug the issues. A binaries built without `ENABLE_FUZZING` options can also be used to reproduce and debug the issues.
