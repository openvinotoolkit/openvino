# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "paddle_tests")

file(GLOB_RECURSE SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
list(FILTER SRC EXCLUDE REGEX standalone_build)

add_executable(${TARGET_NAME} ${SRC})

target_link_libraries(${TARGET_NAME} PRIVATE cnpy frontend_shared_test_classes
        openvino_paddle_frontend openvino::runtime)

add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})

install(TARGETS ${TARGET_NAME}
        RUNTIME DESTINATION tests
        COMPONENT tests
        EXCLUDE_FROM_ALL)

# Test model generating
ov_check_pip_package(REQUIREMENT paddlepaddle
                     MESSAGE_MODE WARNING
                     RESULT_VAR paddlepaddle)

set(TEST_PADDLE_MODELS_DIRNAME test_model_zoo/paddle_test_models)
target_compile_definitions(${TARGET_NAME} PRIVATE -D TEST_PADDLE_MODELS_DIRNAME=\"${TEST_PADDLE_MODELS_DIRNAME}/\")

# If 'paddlepaddle' is not found, code will still be compiled
# but models will not be generated and tests will fail
# This is done this way for 'code style' and check cases - cmake shall pass, but CI machine doesn't need to have
# 'paddlepaddle' installed to check code style
if(paddlepaddle)
    set(TEST_PADDLE_MODELS ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TEST_PADDLE_MODELS_DIRNAME}/)

    file(GLOB_RECURSE PADDLE_ALL_SCRIPTS ${CMAKE_CURRENT_SOURCE_DIR}/*.py)
    set(OUT_FILE ${TEST_PADDLE_MODELS}/generate_done.txt)
    add_custom_command(OUTPUT ${OUT_FILE}
            COMMAND ${PYTHON_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/test_models/gen_wrapper.py
            ${CMAKE_CURRENT_SOURCE_DIR}/test_models/gen_scripts
            ${TEST_PADDLE_MODELS}
            DEPENDS ${PADDLE_ALL_SCRIPTS}
            )
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

add_dependencies(${TARGET_NAME} paddle_test_models)
add_dependencies(${TARGET_NAME} openvino_paddle_frontend)

# Fuzzy tests for PaddlePaddle use IE_CPU engine
if (ENABLE_INTEL_CPU)
    add_dependencies(${TARGET_NAME} openvino_intel_cpu_plugin)
endif()

add_subdirectory(standalone_build)
add_dependencies(${TARGET_NAME} paddle_fe_standalone_build_test)
