# Layer tests

This folder layer tests framework code and test files.

## Getting Started

#### Pre-requisites

* OpenVINO should be configured as usual.

#### Setup

* Set up environment variables for layer tests:
    ```bash
    export MO_ROOT=PATH_TO_MO
    ```
    ```bash
    export PYTHONPATH="path_to_openvino"/tests/layer_tests/:$PYTHONPATH
    ```
* If you need compare scoring results:
    * Set up additional environment variable:
        ```bash
            >export IE_APP_PATH="path_to_IE"
        ```
    * Add IE dependencies in LD_LIBRARY_PATH.

## Run tests
```bash
py.test
```
