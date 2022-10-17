# OpenVINO TensorFlow Frontend tests

There are two types of tests for the TensorFlow Frontend (TF FE): layer tests and unit-tests.

The layer tests are used to validate support of TensorFlow operation by the frontend.

The unit-tests cover TensorFlow format reading functionality, conversion pipeline, and internal transformations
for Transpose Sinking and conversion of sub-graphs with TF FE internal operations into the OpenVINO opset.

## How to build and run unit-tests

For building the TF FE unit-tests, use the CMake target `ov_tensorflow_frontend_tests`. CMake automatically runs
[generation scripts](../tests/test_models/gen_scripts) to create TensorFlow models used in the testing.

Once the build is complete, launch the `ov_tensorflow_frontend_tests` (`ov_tensorflow_frontend_tests.exe` for Windows)
executable file to run all tests for the TensorFlow Frontend. The unit-tests use the GoogleTest framework for execution.

To get a tests coverage report for the TensorFlow Frontend, read the page
on [measuring coverage](../../../../docs/dev/test_coverage.md).

## How to run TensorFlow Frontend layer tests

The layer tests are Python-based and check that a TensorFlow operation is supported by TF FE.
The testing pipeline of the layer tests consists of three steps: create the single layer model with tested operation using TensorFlow,
convert this model into IR by TF FE, infer the original model using TensorFlow, infer IR model using OpenVINO,
and compare the inference results from both frameworks.

The layer tests include two suites for [TensorFlow 1](../../../../tests/tensorflow_tests)
and [TensorFlow 2 Keras](../../../../tests/tensorflow2_keras_tests) operation set support.

To set up environment for running the layer tests, follow these [instructions](../../../../tests/layer_tests/README.md).

To test the whole suite of the TensorFlow 1 operation set support, run the following command:
```bash
py.test tensorflow_tests --use_new_frontend
```

The command line for one operation:
```bash
py.test tensorflow_tests/test_tf_Unique.py --use_new_frontend
```

## See also

 * [OpenVINO TensorFlow Frontend README](../README.md)
 * [OpenVINO™ README](../../../../README.md)
 * [Developer documentation](../../../../docs/dev/index.md)

