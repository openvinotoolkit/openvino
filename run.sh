set -e

build_t=Release
pyenv activate 3128ami
python_api="-DENABLE_PYTHON=OFF -DENABLE_WHEEL=OFF"
js_api="-DENABLE_JS=ON -DCPACK_GENERATOR=NPM -DCMAKE_INSTALL_PREFIX="../src/bindings/js/node/bin""

export PYTHONPATH="/home/amilosze/openvino/bin/intel64/$build_t/python:"$PYTHONPATH
export LD_LIBRARY_PATH="/home/amilosze/openvino/bin/intel64/$build_t/python:"$LD_LIBRARY_PATH
export PYTHONPATH="/home/amilosze/openvino/bin/intel64/$build_t:"$PYTHONPATH  # links .so files needed for some tests
export LD_LIBRARY_PATH="/home/amilosze/openvino/bin/intel64/$build_t:"$LD_LIBRARY_PATH

rm -r build/CMakeFiles/ build/CMakeCache.txt 
cmake -S . -B build -G Ninja -DENABLE_TESTS=ON \
    -DENABLE_TEMPLATE=ON -DENABLE_SAMPLES=OFF -DENABLE_OV_PADDLE_FRONTEND=OFF \
    -DCMAKE_BUILD_TYPE=$build_t $js_api $python_api -DENABLE_INTEL_CPU=ON -DENABLE_INTEL_GPU=OFF -DENABLE_INTEL_NPU=OFF

cmake --build build -j 44
# cmake --build build --target ov_file_load_benchmark -j8
# cmake --build build --target clang_format_fix_all -j8
# cmake --install build --prefix /home/amilosze/ov_install
# cmake --build build --target openvino ov_core_unit_tests openvino_intel_cpu_plugin ov_inference_functional_tests ov_util_tests ov_auto_unit_tests -j 44
# cmake --build build --target ov_ir_frontend_tests -j 44

# release, --privileged. add a flag.  to check out new startegies and types of disks

# rm -r src/bindings/js/node/bin/ src/bindings/js/node/node_modules/ src/bindings/js/node/types/ src/bindings/js/node/dist/
# cmake --install build --prefix "src/bindings/js/node/bin"
# cd src/bindings/js/node && npm i && cd -
# node --test src/bindings/js/node/tests/unit/*.test.js



# ./bin/intel64/"$build_t"/ov_file_load_benchmark
# ./bin/intel64/"$build_t"/ov_util_tests --gtest_filter=*VmPrefetchMappedFileTest*
# ./bin/intel64/"$build_t"/ov_core_unit_tests --gtest_filter=*SharedBufferTest*
./bin/intel64/"$build_t"/ov_core_unit_tests 
# ./bin/intel64/"$build_t"/ov_core_unit_tests

# ./bin/intel64/"$build_t"/base_func_tests
# [  PASSED  ]
# ./bin/intel64/"$build_t"/ov_capi_test --gtest_filter=*ov_core_create_with_config*
# ./bin/intel64/"$build_t"/ov_ir_frontend_tests
# ./bin/intel64/"$build_t"/ov_inference_functional_tests
# ./bin/intel64/"$build_t"/ov_inference_functional_tests --gtest_filter=*CoreBaseTest*
# ./bin/intel64/"$build_t"/ov_onnx_frontend_tests 
# ./bin/intel64/"$build_t"/ov_onnx_frontend_tests --gtest_filter=*up_dir_path*
# ./bin/intel64/"$build_t"/ov_onnx_frontend_tests --gtest_filter='INTERPRETER.onnx_model_expand_scalar_failsafe_node_ort_mem:IE_CPU.onnx_model_expand_scalar_failsafe_node_ort_mem:ONNXLoadTest/FrontEndLoadFromTest.*:ONNX/FrontendLibCloseTest.testModelIsLasDeletedObject/*:ONNX/FrontendLibCloseTest.testPlaceIsLastDeletedObject/*:ONNX/FrontendLibCloseTest.testPlaceFromPlaceIsLastDeletedObject/*:ONNX/FrontendLibCloseTest.testGetVectorOfPlaces/*:OnnxFeMMapReadModel/OnnxFeMmapFixture.*'
./bin/intel64/"$build_t"/ov_util_tests 
# ./bin/intel64/"$build_t"/ov_onnx_frontend_tests 
# ./bin/intel64/"$build_t"/ov_tensorflow_lite_frontend_tests 
# ./bin/intel64/"$build_t"/ov_tensorflow_frontend_tests 
# ./bin/intel64/"$build_t"/ov_paddle_tests


# python -m pytest src/bindings/python/tests -sv
# python -m pytest src/frontends/onnx/tests/tests_python/test_frontendmanager.py::test_register_front_end_path -sv
# ----onnx
# cmake --build build --target ov_onnx_frontend_tests paddle_tests ov_tensorflow_frontend_tests ov_tensorflow_lite_frontend_tests  -j 10
# ./bin/intel64/Debug/ov_onnx_frontend_tests --gtest_filter=*FrontEndLoadFromTest*
# ./bin/intel64/Debug/paddle_tests
# ./bin/intel64/Debug/paddle_tests --gtest_filter="PaddleFuzzyOpTest/FrontEndFuzzyOpTest.testOpFuzzy"
# ./bin/intel64/Debug/ov_tensorflow_lite_frontend_tests 

# ./bin/intel64/Debug/ov_onnx_frontend_tests --gtest_filter="ONNXLoadTest/FrontEndLoadFromTest.load_model_not_exists_at_path*:ONNXLoadTest/FrontEndLoadFromTest.testLoadFromModelProtoUint64_Negative*


# PaddleFuzzyOpTest/FrontEndFuzzyOpTest.testOpFuzzy/paddle_greater_equal_big_int64_greater_equal_big_int64_pdmodel