# Layer tests

This folder layer tests framework code and test files.

## Getting Started

#### Pre-requisites

* OpenVINO should be configured as usual.

#### Setup

* Install requirements:
    ```bash
    pip3 install -r requirements.txt
    ```
* Set up environment variables for layer tests (if you use wheel package path to python api could be removed):
    ```bash
    export PYTHONPATH="path_to_openvino"/tests/layer_tests/:"path_to_openvino"/tools/mo:"path to python api"
    ```
* If there is need to use specific libs it is possible to specify path to them using OV_LIBRARY_PATH env variable
    ```bash
    export OV_LIBRARY_PATH="path_to_libs"
    ```
* To parametrize tests by device and precision (optional)
   ```bash
   export TEST_DEVICE="CPU;GPU"
   export TEST_PRECISION="FP32;FP16"
   ```

## Run tests
```bash
py.test
```
