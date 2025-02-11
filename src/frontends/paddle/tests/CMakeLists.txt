# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "paddle_tests")

# Test model generating
set(PADDLE_REQ "${CMAKE_CURRENT_SOURCE_DIR}/requirements.txt")
if(Python3_Interpreter_FOUND)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/paddle_pip_check.py" ${PADDLE_REQ}
        RESULT_VARIABLE EXIT_CODE
        OUTPUT_VARIABLE OUTPUT_TEXT
        ERROR_VARIABLE ERROR_TEXT)
endif()

if(NOT EXIT_CODE EQUAL 0)
    set(paddlepaddle_FOUND OFF)
    message(WARNING "Python requirement file ${PADDLE_REQ} is not installed, PaddlePaddle testing models weren't generated, some tests will fail due models not found")
else()
    set(paddlepaddle_FOUND ON)
endif()

if(paddlepaddle_FOUND)
    set(ctest_labels OV UNIT)
endif()

ov_add_test_target(
    NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDENCIES
            paddle_test_models
            openvino_paddle_frontend
            paddle_fe_standalone_build_test
        LINK_LIBRARIES
            openvino::cnpy
            frontend_shared_test_classes
            openvino_paddle_frontend
            openvino::runtime
            gtest_main_manifest
            func_test_utils
        ADD_CLANG_FORMAT
        LABELS
            ${ctest_labels} PADDLE_FE
)

set(TEST_PADDLE_MODELS_DIRNAME ${TEST_MODEL_ZOO}/paddle_test_models)
target_compile_definitions(${TARGET_NAME} PRIVATE -D TEST_PADDLE_MODELS_DIRNAME=\"${TEST_PADDLE_MODELS_DIRNAME}/\")

if(paddlepaddle_FOUND)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import paddle; print(paddle.__version__)"
        OUTPUT_VARIABLE PADDLE_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    message(STATUS "PaddlePaddle version: ${PADDLE_VERSION}")
    
    if(PADDLE_VERSION VERSION_GREATER_EQUAL "2.6.0" OR PADDLE_VERSION VERSION_EQUAL "0.0.0")
        set(PADDLEDET_OPS_URL "https://raw.githubusercontent.com/PaddlePaddle/PaddleDetection/release/2.5/ppdet/modeling/ops.py")
        set(PADDLEDET_OPS_SHA256 "e3da816421698ee97bb272c4410a03c300ab92045b7c87cccb9e52a8c18bc088")
        set(PADDLEDET_DIRNAME ${CMAKE_CURRENT_BINARY_DIR}/thirdparty/PaddleDetection/release25/ppdet/modeling/)
    else()
        set(PADDLEDET_OPS_URL "https://raw.githubusercontent.com/PaddlePaddle/PaddleDetection/release/2.1/ppdet/modeling/ops.py")
        set(PADDLEDET_OPS_SHA256 "5cc079eda295ed78b58fba8223c51d85a931a7069ecad51c6af5c2fd26b7a8cb")
        set(PADDLEDET_DIRNAME ${CMAKE_CURRENT_BINARY_DIR}/thirdparty/PaddleDetection/release21/ppdet/modeling/)
    endif()
    
    DownloadAndCheck(${PADDLEDET_OPS_URL} ${PADDLEDET_DIRNAME}/ops.py PADDLEDET_FATAL PADDLEDET_RESULT ${PADDLEDET_OPS_SHA256})
endif()

# If 'paddlepaddle' is not found, code will still be compiled, but models will not be generated and tests will fail
# This is done this way for 'code style' and check cases - cmake shall pass, but CI machine doesn't need to have
# 'paddlepaddle' installed to check code style
if(PADDLEDET_RESULT)
    set(TEST_PADDLE_MODELS ${TEST_MODEL_ZOO_OUTPUT_DIR}/paddle_test_models/)

    file(GLOB_RECURSE PADDLE_ALL_SCRIPTS ${CMAKE_CURRENT_SOURCE_DIR}/*.py)
    set(OUT_FILE ${TEST_PADDLE_MODELS}/generate_done.txt)
    add_custom_command(OUTPUT ${OUT_FILE}
            COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${PADDLEDET_DIRNAME}
                ${Python3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/test_models/gen_wrapper.py
                    ${CMAKE_CURRENT_SOURCE_DIR}/test_models/gen_scripts
                    ${TEST_PADDLE_MODELS}
            DEPENDS ${PADDLE_ALL_SCRIPTS})
    add_custom_target(paddle_test_models DEPENDS ${OUT_FILE})

    install(DIRECTORY ${TEST_PADDLE_MODELS}
            DESTINATION tests/${TEST_PADDLE_MODELS_DIRNAME}
            COMPONENT tests
            EXCLUDE_FROM_ALL)

else()
    # Produce warning message at build time as well
    add_custom_command(OUTPUT unable_build_paddle_models.txt
            COMMAND ${CMAKE_COMMAND}
            -E cmake_echo_color --red "Warning: Unable to generate PaddlePaddle test models. Running '${TARGET_NAME}' will likely fail"
            )
    add_custom_target(paddle_test_models DEPENDS unable_build_paddle_models.txt)
endif()

# Fuzzy tests for PaddlePaddle use IE_CPU engine
if(ENABLE_INTEL_CPU)
    add_dependencies(${TARGET_NAME} openvino_intel_cpu_plugin)
endif()

add_subdirectory(standalone_build)
