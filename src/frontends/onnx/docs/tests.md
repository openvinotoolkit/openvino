# ONNX Frontend tests

## ONNX Frontend testing places
- [C++ gtest-based tests](../tests)
- [Python frontend tests](../../../../src/bindings/python/tests/test_frontend)
- [Python operators tests](../../../../src/bindings/python/tests/test_onnx)
- [Python tests to confirm operators compliance with the ONNX standard](../../../../src/bindings/python/tests/test_onnx/test_backend.py)
- [Python OpenModelZoo tests](../../../../src/bindings/python/tests/test_onnx/test_zoo_models.py)
- [OpenVINO™ Execution Provider for ONNX Runtime tests](../../../../.ci/azure/linux_onnxruntime.yml)


## How to run tests locally
### C++ tests
1. Build OpenVINO with `-DENABLE_TESTS=ON` flag.
2. Run `openvino/bin/intel64/<BUILD_TYPE>/ov_onnx_frontend_tests`, where `BUILD_TYPE` can be `Debug` or `Release`. It depends on `-DCMAKE_BUILD_TYPE`.
You can filter tests using `--gtest_filter` flag like any others gtest-based tests.

Example command:
```
/home/user_name/openvino/bin/intel64/Release/ov_onnx_frontend_tests --gtest_filter=**

```


### PRE steps for all Python tests:
1. Build OpenVINO with with `-DENABLE_PYTHON=ON`, preferably in a `Python` virtual environment. In order to avoid problems with many Python interpreters installed, it is worth to set also `-DPYTHON_EXECUTABLE=`which python3`` build flag.
Note: if you want to run tests from installation directory location (like in the CI), you should add `-P cmake_install.cmake` and `-DCOMPONENT=tests` flags and install OpenVINO (via `cmake --build . --target install`) as additional steps.
2. Set-up `Python` paths via `source <OV_INSTALL_DIR>/setupvars.sh` (or `sh <INSTALL_DIR>\setupvars.bat` for `Windows`) or set `export PYTHONPATH=<OV_REPO_PATH>/bin/intel64/<BUILD_TYPE>/python_api/python<PYTHON_VERSION>` (note that `set` instead of `export` should be used for `Windows`).
3. Instal `Python` dependencies:
```
pip install -r openvino/src/bindings/python/requirements.txt```
pip install -r openvino/src/bindings/python/requirements_test.txt
```


### Python frontend tests
Run ONNX Frontend Python tests using command:
a) for build layout run:
```
pytest openvino/src/bindings/python/tests/test_frontend/test_frontend_onnx*
```
b) for install layout run:
```
pytest <OV_INSTALL_DIR>/tests/pyopenvino/tests/test_frontend/test_frontend_onnx*
```


### Python ONNX operators tests
Run ONNX operators Python tests using command:
a) for build layout run:
```
pytest openvino/src/bindings/python/tests/test_onnx \
    --ignore=openvino/src/bindings/python/tests/test_onnx/test_zoo_models.py \
    --ignore=openvino/src/bindings/python/tests/test_onnx/test_backend.py
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
pytest openvino/src/bindings/python/tests/test_onnx/test_backend.py -sv -k 'not cuda'
```
b) for install layout run:
```
pytest <OV_INSTALL_DIR>/tests/pyopenvino/tests/test_onnx/test_backend.py -sv -k 'not cuda'
```


### Python OpenModelZoo tests
PRE: Download models from https://github.com/onnx/models (TODO LINK) (the current OV version uses https://github.com/onnx/models/commit/d58213534f2a4d1c4b19ba62b3bb5f544353256e version) or use `openvino/src/bindings/python/tests/test_onnx/model_zoo_preprocess.sh` script to realize it. Note, that the second approach is preferable, because some test data require preprocessing.
```
model_zoo_preprocess.sh -d <ONNX_MODELS_DIR> -o
```
Note that the current CI test also models from MSFT package (`-m` flag of `model_zoo_preprocess.sh` script), but it is deprecated pipeline. If you achieve some problems with it, please contact [openvino-onnx-frontend-maintainers](https://github.com/orgs/openvinotoolkit/teams/openvino-onnx-frontend-maintainers).

a) for build layout run:
```
pytest --backend=CPU <REPO_DIR>/src/bindings/python/tests/test_onnx/test_zoo_models.py -v -n 4 --forked -k 'not _cuda' --model_zoo_dir=/shared/onnx_model_zoo/
```

Note that you can run tests also only for a single model. It can look like:
```
pytest --backend=CPU <REPO_DIR>/src/bindings/python/tests/test_onnx/test_zoo_models.py -v -n 4 --forked -k 'not _cuda' --model_zoo_dir=/shared/onnx_model_zoo/ -k test_onnx_model_zoo_vision_classification_alexnet_model_bvlcalexnet_9_bvlc_alexnet_model_cpu
```
b) for install layout run:
```
pytest --backend=CPU <INSTALL_DIR>/tests/pyopenvino/tests/test_onnx/test_zoo_models.py -v -n 4 --forked -k 'not _cuda' --model_zoo_dir=/shared/onnx_model_zoo/
```
Be aware that a model model is tested in two stages: importing (`OnnxBackendModelImportTest`) and inference (`OnnxBackendModelExecutionTest`).

Tip: Very useful `pytest` flag during working with ONNX tests is `--collect-only`. It's allow to list all possible tests (based on available models) without running them.
Example output of such command can look like:
```
<TestCaseFunction test_onnx_model_zoo_vision_style_transfer_fast_neural_style_model_rain_princess_8_rain_princess_model_cpu>
<TestCaseFunction test_onnx_model_zoo_vision_style_transfer_fast_neural_style_model_rain_princess_9_rain_princess_rain_princess_cpu>
<TestCaseFunction test_onnx_model_zoo_vision_style_transfer_fast_neural_style_model_udnie_8_udnie_udnie_cpu
...

```


### OpenVINO™ Execution Provider for ONNX Runtime tests
1. Clone ONNX Runtime repo (version `rel-1.8.1` is used in the current CI configuration):
```
git clone --branch rel-1.8.1 --single-branch --recursive https://github.com/microsoft/onnxruntime.git
```
1a. Additional step required for `rel-1.8.1` version:
```
cd <OV_INSTALL_DIR>/deployment_tools/inference_engine && touch version.txt && echo "2021.4" > version.txt
```
2. Set-up OpenVINO environment:
```
source <OV_INSTALL_DIR>/setupvars.sh
```
3. Build ONNX Runtime with openvino enabled:
```
./build.sh --config RelWithDebInfo --use_openvino CPU_FP32 --build_shared_lib --parallel
```
After building finish the tests are run automatically (if `--skip_tests` flag is not set).
Note that ONNX Runtime requires (the version from the current master) cmake version greater than `3.25`. You can parametrize `build.sh` script with your custom cmake path using `--cmake_path` argument.
