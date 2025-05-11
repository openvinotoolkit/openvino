# Layer Tests

The layer tests primarily aim to validate support for PyTorch, TensorFlow, TensorFlow Lite, and ONNX frameworks' operations by OpenVINO.
The test pipeline includes the following steps:
1. Creation of a model with the tested operation using original framework API
2. Conversion of the created model using OpenVINO's `convert_model` method
3. Inference of both the original and converted models using the framework and OpenVINO on random input data
4. Checking whether the inference results from OpenVINO and the framework are the same or different within a tolerance threshold

## Setup Environment

To set up the environment for launching layer tests, perform the following steps:
1. Install the OpenVINO wheel package. If you're testing changes in OpenVINO, build your local wheel package for installation.
Find instructions on how to build on [wiki page](https://github.com/openvinotoolkit/openvino/blob/master/docs/dev/build.md).
    ```sh
    pip install openvino.whl
    ```
2. (Optional) Install the OpenVINO Tokenizers wheel package if you're testing the support of operations using conversion and operation extensions from OpenVINO Tokenizers.
If you're testing changes in OpenVINO Tokenizers, build your local wheel package for installation.
Find instructions on how to build on [GitHub page](https://github.com/openvinotoolkit/openvino_tokenizers)
    ```sh
    pip install openvino_tokenizers.whl
    ```
3. Install requirements for running layer tests.
    ```sh
    cd tests/layer_tests
    pip install -r requirements.txt
    ```

## Run Tests

Set environment variables `TEST_DEVICE` and `TEST_PRECISION` to select device and inference precision for OpenVINO inference. Allowed values for `TEST_DEVICE` are `CPU` and `GPU`. Allowed values for `TEST_PRECISION` are `FP32` and `FP16`.

Example to run the TensorFlow layer test for the `tf.raw_ops.Unique` operation on CPU with default inference precision for device:
   ```sh
   cd tests/layer_tests
   export TEST_DEVICE="CPU"
   pytest tensorflow_tests/test_tf_Unique.py
   ```

Example to run the PyTorch layer test for the `torch.linalg.cross` operation on CPU and GPU with `FP16` and `FP32` inference precisions:
   ```sh
   cd tests/layer_tests
   export TEST_DEVICE="CPU;GPU"
   export TEST_PRECISION="FP32;FP16"
   pytest pytorch_tests/test_cross.py
   ```
