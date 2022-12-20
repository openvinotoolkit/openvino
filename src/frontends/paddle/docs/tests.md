# OpenVINO Paddle Frontend Tests

OpenVINO Paddle Frontend [tests](../tests) have the following structure:
 * `test_models/gen_scripts/` - Python script to generate PaddlePaddle test models with handy helpers.
 * `standalone_build` - PaddlePaddle frontend standalone build test.
 * a couple of files - PaddlePaddle frontend operator set unit-test framework and Paddle frontend API unit-test written using Google Test.

## How to build and run tests

OpenVINO Paddle Frontend is covered by tests from the binary `paddle_tests`. This binary can be built by the target with the same name.

## How to add a unit-test
1. Implement python script in `test_models/gen_scripts/` to get the PaddlePaddle test model and the reference inputs and outputs from PaddlePaddle. 
2. Register the test case name to `tests/src/op_fuzzy.cpp`, which is part of the operator set unit-test framework.

## See also
 * [OpenVINO™ Paddle Frontend README](../README.md)
