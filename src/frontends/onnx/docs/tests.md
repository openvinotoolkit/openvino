# ONNX Frontend tests

## ONNX Frontend testing places
- [C++ gtest-based tests](../tests)
- [Python frontend tests](../../../../src/bindings/python/tests/test_frontend)
- [Python operators tests](../../../../src/bindings/python/tests/test_onnx)
- [Python tests to confirm operator compliance with the ONNX standard](../../../../src/bindings/python/tests/test_onnx/test_backend.py)
- [Python OpenModelZoo tests](../../../../src/bindings/python/tests/test_onnx/test_zoo_models.py)
- [Tests for OpenVINO™ Execution Provider with ONNX Runtime](../../../../.ci/azure/linux_onnxruntime.yml)


## How to run the tests locally
## C++ tests
1. Install Python dependencies:
```
pip install -r <OV_REPO_DIR>/src/frontends/onnx/tests/requirements.txt
```
After that CMake will produce test models from existin [*.prototxt files](../tests/models)
2. Build OpenVINO with the `-DENABLE_TESTS=ON` flag.
3. Run `<OV_REPO_DIR>/bin/intel64/<OV_BUILD_TYPE>/ov_onnx_frontend_tests`, where <OV_REPO_DIR> is your workspace folder, and `OV_BUILD_TYPE` can be `Debug` or `Release`, depending on the `-DCMAKE_BUILD_TYPE` CMake build option.
You can filter tests using the `--gtest_filter` flag like any other gtest-based tests.

For example:
```
<OV_REPO_DIR>/bin/intel64/<OV_BUILD_TYPE>/ov_onnx_frontend_tests --gtest_filter=*add*

```


## Pre-steps for all Python tests
1. Build OpenVINO with `-DENABLE_PYTHON=ON`, preferably in a `Python` virtual environment. To avoid problems with too many Python interpreters installed on the host, you can also set the `-DPython3_EXECUTABLE=<PYTHON_INTERPRETER_PATH>` build option (requires cmake 3.16 and higher).
> **NOTE**: If you want to run the tests from the installation directory (like in the CI), add the `-P cmake_install.cmake` and `-DCOMPONENT=tests` CMake build options, and install OpenVINO via `cmake --build . --target install` as additional steps.
2. Set up Python paths via `source <OV_INSTALL_DIR>/setupvars.sh` for Linux, `. <path-to-setupvars-folder>/setupvars.ps1` for Windows PowerShell, or `sh <INSTALL_DIR>\setupvars.bat` for Windows Command Prompt.
3. Install Python dependencies:
```
pip install -r <OV_REPO_DIR>/src/bindings/python/requirements.txt
pip install -r <OV_REPO_DIR>/src/bindings/python/requirements_test.txt
```


## Python frontend tests
You can run ONNX Frontend Python tests using the following commands:

- For the build layout:
```
pytest <OV_REPO_DIR>/src/bindings/python/tests/test_frontend/test_frontend_onnx*
```
- For the installation layout:
```
pytest <OV_INSTALL_DIR>/tests/pyopenvino/tests/test_frontend/test_frontend_onnx*
```


### Python ONNX operators tests
You can run ONNX operators Python tests using the following commands:

- For the build layout:
```
pytest <OV_REPO_DIR>/src/bindings/python/tests/test_onnx \
    --ignore=<OV_REPO_DIR>/src/bindings/python/tests/test_onnx/test_zoo_models.py \
    --ignore=<OV_REPO_DIR>/src/bindings/python/tests/test_onnx/test_backend.py
```
- For the installation layout:
```
pytest <OV_INSTALL_DIR>/tests/pyopenvino/tests/test_onnx \
    --ignore=<OV_INSTALL_DIR>/tests/pyopenvino/tests/test_onnx/test_zoo_models.py \
    --ignore=<OV_INSTALL_DIR>/tests/pyopenvino/tests/test_onnx/test_backend.py
```


### Python tests to confirm operator compliance with the ONNX standard
- For the build layout:
```
pytest <OV_REPO_DIR>/src/bindings/python/tests/test_onnx/test_backend.py -sv -k 'not cuda'
```
- For the installation layout:
```
pytest <OV_INSTALL_DIR>/tests/pyopenvino/tests/test_onnx/test_backend.py -sv -k 'not cuda'
```


## Python OpenModelZoo tests
Preparation: Download [models](https://github.com/onnx/models). The current OpenVINO ONNX Frontend uses [this version](https://github.com/onnx/models/commit/d58213534f2a4d1c4b19ba62b3bb5f544353256e) or [the model_zoo_preprocess.sh script](../../../../src/bindings/python/tests/test_onnx/model_zoo_preprocess.sh). The second approach is preferable, because some test data requires preprocessing.
```
model_zoo_preprocess.sh -d <ONNX_MODELS_DIR> -o
```
Note that the current CI also tests models from the MSFT package (the `-m` flag of the `model_zoo_preprocess.sh` script can enable the processing of it), but it is a deprecated pipeline. If you encounter problems with this package, contact [openvino-onnx-frontend-maintainers](https://github.com/orgs/openvinotoolkit/teams/openvino-onnx-frontend-maintainers).

Commands to run OpenModelZoo tests:

- For the build layout:
```
pytest --backend=CPU <OV_REPO_DIR>/src/bindings/python/tests/test_onnx/test_zoo_models.py -v -n 4 --forked -k 'not _cuda' --model_zoo_dir=<ONNX_MODELS_DIR>
```

Note that you can also run tests for a single model only, for example:
```
pytest --backend=CPU <OV_REPO_DIR>/src/bindings/python/tests/test_onnx/test_zoo_models.py -v -n 4 --forked -k 'not _cuda' --model_zoo_dir=<ONNX_MODELS_DIR> -k test_onnx_model_zoo_vision_classification_alexnet_model_bvlcalexnet_9_bvlc_alexnet_model_cpu
```
- For the installation layout:
```
pytest --backend=CPU <OV_INSTALL_DIR>/tests/pyopenvino/tests/test_onnx/test_zoo_models.py -v -n 4 --forked -k 'not _cuda' --model_zoo_dir=<ONNX_MODELS_DIR>
```
Be aware that each model is tested in two stages: importing (`OnnxBackendModelImportTest`) and inference (`OnnxBackendModelExecutionTest`).

> **TIP**: A useful `pytest` flag during working with OpenModelZoo tests is `--collect-only`. It enables you to list all the possible tests (based on available models) without running them.
An example output:
```
<TestCaseFunction test_onnx_model_zoo_vision_style_transfer_fast_neural_style_model_rain_princess_8_rain_princess_model_cpu>
<TestCaseFunction test_onnx_model_zoo_vision_style_transfer_fast_neural_style_model_rain_princess_9_rain_princess_rain_princess_cpu>
...

```


## Tests for OpenVINO Execution Provider with ONNX Runtime
1. Clone the ONNX Runtime repository (the `rel-1.8.1` version is used in the current CI configuration):
```
git clone --branch rel-1.8.1 --single-branch --recursive https://github.com/microsoft/onnxruntime.git
```
- (Optional) To build ONNX Runtime with a non-release OpenVINO version, an additional step is required for the `rel-1.8.1` version:
```
cd <OV_INSTALL_DIR>/deployment_tools/openvino && touch version.txt && echo "2021.4" > version.txt
```
2. Set up the OpenVINO environment:
```
source <OV_INSTALL_DIR>/setupvars.sh
```
3. Build ONNX Runtime with OpenVINO enabled:
```
<ONNX_RUNTIME_REPO_DIR>/build.sh --config RelWithDebInfo --use_openvino CPU_FP32 --build_shared_lib --parallel
```
After the building is finished, the tests will run automatically if the `--skip_tests` flag is not set.
Note that ONNX Runtime version from the master branch requires CMake version being greater than `3.25`. You can parametrize the `build.sh` script with your custom CMake version by passing the path via the `--cmake_path` argument.
