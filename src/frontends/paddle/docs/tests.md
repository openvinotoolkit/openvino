# OpenVINO Paddle Frontend Tests

OpenVINO Paddle Frontend is covered by tests from the binary `paddle_tests`. This binary can be built by the target with the same name.

OpenVINO Paddle Frontend [tests](../tests/) have next structure:
 * `test_models/gen_scripts/` - The python script to generate PaddlePaddle test models with handy helpers.
 * `src` - PaddlePaddle frontend tests suite written using Google Test.


OpenVINO Paddle Frontend Unit Test development flow:
1. Implement python script in `test_models/gen_scripts/` to get PaddlePaddle test model and the reference inputs and outputs from PaddlePaddle. 
2. Register the test case name to `tests/src/op_fuzzy.cpp`.

## See also
 * [OpenVINO™ Paddle Frontend README](../README.md)
