# ONNX Frontend tests

## ONNX Frontend testing places
- [C++ gtest-based tests](../tests)
- [Python frontend tests](../../../../src/bindings/python/tests/test_frontend)
- [Python operators tests](../../../../src/bindings/python/tests/test_onnx)
- [Python tests to confirm operators compliance with the ONNX standard](../../../../src/bindings/python/tests/test_onnx/test_backend.py)
- [Python OpenModelZoo tests](../../../../src/bindings/python/tests/test_onnx/test_zoo_models.py)
- [OpenVINO™ Execution Provider for ONNX Runtime tests](../../../../.ci/azure/linux_onnxruntime.yml)


## How to run the tests locally
## C++ tests
1. Build OpenVINO with `-DENABLE_TESTS=ON` flag.
2. Run `<OV_REPO_DIR>/bin/intel64/<OV_BUILD_TYPE>/ov_onnx_frontend_tests`, where <OV_REPO_DIR> is your workspace folder, `OV_BUILD_TYPE` can be `Debug` or `Release`. It depends on `-DCMAKE_BUILD_TYPE` cmake build option.
You can filter tests using `--gtest_filter` flag like any other gtest-based tests.

Example command:
```
<OV_REPO_DIR>/bin/intel64/<OV_BUILD_TYPE>/ov_onnx_frontend_tests --gtest_filter=*add*

```


## Pre-steps for all Python tests:
1. Build OpenVINO with `-DENABLE_PYTHON=ON`, preferably in a `Python` virtual environment. In order to avoid problems with many Python interpreters installed on the host, it is worth to set also `-DPYTHON_EXECUTABLE=PYTHON_INTERPRETER_PATH` build option.
Note: if you want to run the tests from installation directory location (like in the CI), you should add `-P cmake_install.cmake` and `-DCOMPONENT=tests` cmake build options and install OpenVINO (via `cmake --build . --target install`) as additional steps.
2. Set-up `Python` paths via `source <OV_INSTALL_DIR>/setupvars.sh` (or `sh <INSTALL_DIR>\setupvars.bat` for `Windows`) or set `export PYTHONPATH=<OV_REPO_DIR>/bin/intel64/<BUILD_TYPE>/python_api/python<PYTHON_VERSION>` (note that `set` instead of `export` should be used for `Windows`).
3. Instal `Python` dependencies:
```
pip install -r <OV_REPO_DIR>/src/bindings/python/requirements.txt
pip install -r <OV_REPO_DIR>/src/bindings/python/requirements_test.txt
```


## Python frontend tests
Run ONNX Frontend Python tests using commands:

a) for build layout run:
```
pytest <OV_REPO_DIR>/src/bindings/python/tests/test_frontend/test_frontend_onnx*
```
b) for install layout run:
```
pytest <OV_INSTALL_DIR>/tests/pyopenvino/tests/test_frontend/test_frontend_onnx*
```


### Python ONNX operators tests
Run ONNX operators Python tests using commands:

a) for build layout run:
```
pytest <OV_REPO_DIR>/src/bindings/python/tests/test_onnx \
    --ignore=<OV_REPO_DIR>/src/bindings/python/tests/test_onnx/test_zoo_models.py \
    --ignore=<OV_REPO_DIR>/src/bindings/python/tests/test_onnx/test_backend.py
```
b) for install layout run:
```
pytest <OV_INSTALL_DIR>/tests/pyopenvino/tests/test_onnx \
    --ignore=<OV_INSTALL_DIR>/tests/pyopenvino/tests/test_onnx/test_zoo_models.py \
    --ignore=<OV_INSTALL_DIR>/tests/pyopenvino/tests/test_onnx/test_backend.py
```


### Python tests to confirm operators compliance with the ONNX standard
a) for build layout run:
```
pytest <OV_REPO_DIR>/src/bindings/python/tests/test_onnx/test_backend.py -sv -k 'not cuda'
```
b) for install layout run:
```
pytest <OV_INSTALL_DIR>/tests/pyopenvino/tests/test_onnx/test_backend.py -sv -k 'not cuda'
```


## Python OpenModelZoo tests
Pre: Download [models](https://github.com/onnx/models) (the current OpenVINO ONNX Frontend version uses [d58213534f2a4d1c4b19ba62b3bb5f544353256e](https://github.com/onnx/models/commit/d58213534f2a4d1c4b19ba62b3bb5f544353256e) version) or use [model_zoo_preprocess.sh](../../../../src/bindings/python/tests/test_onnx/model_zoo_preprocess.sh) script. The second approach is preferable, because some test data requires preprocessing.
```
model_zoo_preprocess.sh -d <ONNX_MODELS_DIR> -o
```
Note that the current CI test also models from MSFT package (`-m` flag of `model_zoo_preprocess.sh` script enables processing it), but it is a deprecated pipeline. If you encounter some problems with this package, please contact [openvino-onnx-frontend-maintainers](https://github.com/orgs/openvinotoolkit/teams/openvino-onnx-frontend-maintainers).

Command to run OpenModelZoo tests:

a) for build layout run:
```
pytest --backend=CPU <OV_REPO_DIR>/src/bindings/python/tests/test_onnx/test_zoo_models.py -v -n 4 --forked -k 'not _cuda' --model_zoo_dir=<ONNX_MODELS_DIR>
```

Note that you can run tests also only for a single model. The command in such a case can look like:
```
pytest --backend=CPU <OV_REPO_DIR>/src/bindings/python/tests/test_onnx/test_zoo_models.py -v -n 4 --forked -k 'not _cuda' --model_zoo_dir=<ONNX_MODELS_DIR> -k test_onnx_model_zoo_vision_classification_alexnet_model_bvlcalexnet_9_bvlc_alexnet_model_cpu
```
b) for install layout run:
```
pytest --backend=CPU <OV_INSTALL_DIR>/tests/pyopenvino/tests/test_onnx/test_zoo_models.py -v -n 4 --forked -k 'not _cuda' --model_zoo_dir=<ONNX_MODELS_DIR>
```
Be aware that each model is tested in two stages: importing (`OnnxBackendModelImportTest`) and inference (`OnnxBackendModelExecutionTest`).

Tip: Very useful `pytest` flag during working with OpenModelZoo tests is `--collect-only`. It allows to list all possible tests (based on available models) without running them.
Example output of such command can look like:
```
<TestCaseFunction test_onnx_model_zoo_vision_style_transfer_fast_neural_style_model_rain_princess_8_rain_princess_model_cpu>
<TestCaseFunction test_onnx_model_zoo_vision_style_transfer_fast_neural_style_model_rain_princess_9_rain_princess_rain_princess_cpu>
...

```


## OpenVINO™ Execution Provider for ONNX Runtime tests
1. Clone ONNX Runtime repo (the version `rel-1.8.1` is used in the current CI configuration):
```
git clone --branch rel-1.8.1 --single-branch --recursive https://github.com/microsoft/onnxruntime.git
```
(Optional) Additional step required for `rel-1.8.1` version (it allows to build ONNX Runtime with NOT release OpenVINO version):
```
cd <OV_INSTALL_DIR>/deployment_tools/inference_engine && touch version.txt && echo "2021.4" > version.txt
```
2. Set-up OpenVINO environment:
```
source <OV_INSTALL_DIR>/setupvars.sh
```
3. Build ONNX Runtime with openvino enabled:
```
<ONNX_RUNTIME_REPO_DIR>/build.sh --config RelWithDebInfo --use_openvino CPU_FP32 --build_shared_lib --parallel
```
After building finish the tests are run automatically (if `--skip_tests` flag is not set).
Note that ONNX Runtime version from the master branch requires cmake version greater than `3.25`. You can parametrize `build.sh` script with your custom cmake version by passing path via `--cmake_path` argument.
