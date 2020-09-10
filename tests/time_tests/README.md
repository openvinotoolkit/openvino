# Time Tests

This test suite contains pipelines measured for time of execution.

## Getting Started

Each pipeline is an executable. Time executing it is measured. Pipelines may
also measure its parts.

## Pre-requisites

To build time tests you need Inference Engine develooper package.

## Measuring time

1. Build tests
``` bash
cmake .. -DInferenceEngineDeveloperPackage_DIR=../../../build && make time-tests
```
2. Run test
``` bash
../../../bin/intel64/Release/infer-pipeline -m model.xml -d CPU
```

