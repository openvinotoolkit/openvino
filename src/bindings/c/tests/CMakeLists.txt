# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# OpenVINO Legacy C API test sample
set(TARGET_NAME "InferenceEngineCAPITests")

add_executable(${TARGET_NAME} ie_c_api_test.cpp test_model_repo.hpp test_model_repo.cpp)

target_link_libraries(${TARGET_NAME} PRIVATE openvino_c common_test_utils gtest_main)

if(ENABLE_AUTO OR ENABLE_MULTI)
    add_dependencies(${TARGET_NAME} openvino_auto_plugin)
endif()

add_dependencies(${TARGET_NAME} mock_engine)

add_cpplint_target(${TARGET_NAME}_cpplint FOR_TARGETS ${TARGET_NAME})

install(TARGETS ${TARGET_NAME}
    RUNTIME DESTINATION tests
    COMPONENT tests
    EXCLUDE_FROM_ALL)

# OpenVINO 2.0 and Legacy C API test sample
set(TARGET_NAME "ov_capi_test")

file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/ov_*.cpp test_model_repo.cpp)
file(GLOB HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

if(NOT TARGET OpenCL::OpenCL)
    list(FILTER SOURCES EXCLUDE REGEX ov_remote_context_test.cpp)
endif()

add_executable(${TARGET_NAME} ${SOURCES} ${HEADERS})
target_link_libraries(${TARGET_NAME} PRIVATE openvino_c openvino::util
    common_test_utils gtest_main)

target_include_directories(${TARGET_NAME} PUBLIC
    $<BUILD_INTERFACE:${OPENVINO_API_SOURCE_DIR}/include>)

if(TARGET OpenCL::OpenCL)
    target_link_libraries(${TARGET_NAME} PRIVATE OpenCL::OpenCL)
endif()

add_dependencies(${TARGET_NAME} mock_engine)

ov_add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})

install(TARGETS ${TARGET_NAME}
    RUNTIME DESTINATION tests
    COMPONENT tests
    EXCLUDE_FROM_ALL)
