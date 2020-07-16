# Fuzzing howto {#openvino_docs_HOWTO_fuzzing_HOWTO}

## Intended Audience

This document is for a developer who wants to contribute fuzz tests.

## Purpose

This document walks you through creating your first fuzzer, running it and evaluating its quality.

## Prerequisites

- Linux OS or Mac OS.

- [American Fuzzy Loop](http://lcamtuf.coredump.cx/afl/) if building   with GCC.

## Steps

1. Create a fuzz test in the existing project at `./tests/fuzz`. Fuzz test must
   follow `<test name>-fuzzer.cc` naming scheme and implement a
   `LLVMFuzzerTestOneInput` entry point.

``` bash
cat << EOF > ./tests/fuzz/test_name-fuzzer.cc
#include <stdint.h>
#include <cstdlib>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  // put your fuzzing code here and use data+size as input.
  return 0;  // always return 0
}
EOF
```

2. Implement test logic under `LLVMFuzzerTestOneInput`.

See example fuzz test at `tests/fuzz/read_network-fuzzer.cc`.

3. Build fuzz tests with `-DENABLE_FUZZING=ON` flag for cmake.

``` bash
    mkdir -p build && \
    (cd build && \
    CXX=afl-g++ CC=afl-gcc cmake -DCMAKE_BUILD_TYPE=Debug -DENABLE_FUZZING=ON -DENABLE_TESTS=ON .. && \
    make fuzz --jobs=$(getconf _NPROCESSORS_ONLN))
```

4. Prepare sample inputs for your fuzz test to teach fuzzer engine on input
   structure

``` bash
(cd bin/intel64/Debug && \
mkdir test_name-corpus && \
echo sample input > test_name-corpus/in1.txt)
```

5. Evaluate fuzz test with `afl-fuzz` fuzzing engine

Run fuzz test:

``` bash
(cd bin/intel64/Debug && \
afl-fuzz -i test_name-corpus -o test_name-out -- ./test_name-fuzzer @@
```

While fuzz test is running it prints out statistics. Besides just crashes `uniq
crashes` and hangs `uniq hangs` you should care about fuzz test quality:

- Fuzz test should be fast - speed of execution `exec speed` should be at least
  100 exec/s. Speed less than 20 exec/s is not acceptable.

- Fuzz test should be able to explore new code paths `map coverage` and
  `findings in depth`. Confirm it is increasing while fuzz test is running.

6. Reproduce fuzz test findings

All issues found by fuzz test are stored as a file in output folder specified
earlier via `-o` afl-fuzz option. To reproduce an issue run fuzz test executable
with an issue file as an argument.

## Summary

We have created a simple fuzz test, run it and asses its results.

## Extension

Try run parallel fuzzing with the help of
[afl-utils](https://gitlab.com/rc0r/afl-utils).

## Tips or FAQs

GCC 7 in Ubuntu 18.04 LTS has a
[defect](https://bugs.launchpad.net/ubuntu/+source/afl/+bug/1774816). Upgrade
GCC 7 for AFL to work. GCC version `Ubuntu 7.3.0-27ubuntu1~18.04` works OK.
