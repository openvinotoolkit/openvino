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
* Set up environment variables for layer tests:
    ```bash
    export MO_ROOT=PATH_TO_MO
    ```
    ```bash
    export PYTHONPATH="path_to_openvino"/tests/layer_tests/:$PYTHONPATH
    ```
    ```bash
    export IE_APP_PATH="path_to_IE"
    ```
* Add IE dependencies in LD_LIBRARY_PATH.

## Run tests
```bash
py.test
```
