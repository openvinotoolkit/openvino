# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "onnx_fe_lib_close_tests")

file(GLOB SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_executable(${TARGET_NAME} ${SRC})
target_link_libraries(${TARGET_NAME} PRIVATE frontend_shared_test_classes openvino_onnx_frontend frontend_common)
target_compile_definitions(${TARGET_NAME} PRIVATE -D TEST_ONNX_MODELS_DIRNAME=\"${TEST_ONNX_MODELS_DIRNAME}/\")
target_compile_definitions(${TARGET_NAME} PRIVATE -D MANIFEST=\"${MANIFEST}\")

add_dependencies(${TARGET_NAME} openvino_onnx_frontend test_model_zoo)

add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})

install(TARGETS ${TARGET_NAME}
        RUNTIME DESTINATION tests
        COMPONENT tests
        EXCLUDE_FROM_ALL)
