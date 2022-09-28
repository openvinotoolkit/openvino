# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# OpenVINO Legacy C API test sample
set(TARGET_NAME "InferenceEngineCAPITests")

add_executable(${TARGET_NAME} ie_c_api_test.cpp test_model_repo.hpp)

target_link_libraries(${TARGET_NAME} PRIVATE openvino_c commonTestUtils gtest_main)

target_compile_definitions(${TARGET_NAME}
    PRIVATE
    $<$<BOOL:${ENABLE_GAPI_PREPROCESSING}>:ENABLE_GAPI_PREPROCESSING>
    DATA_PATH=\"${DATA_PATH}\"
    MODELS_PATH=\"${MODELS_PATH}\")

if(ENABLE_AUTO OR ENABLE_MULTI)
    add_dependencies(${TARGET_NAME} openvino_auto_plugin)
endif()

if(ENABLE_AUTO_BATCH)
    add_dependencies(${TARGET_NAME} openvino_auto_batch_plugin)
endif()

if(ENABLE_INTEL_CPU)
    add_dependencies(${TARGET_NAME} openvino_intel_cpu_plugin)
endif()

if(ENABLE_INTEL_GPU)
    add_dependencies(${TARGET_NAME} openvino_intel_gpu_plugin)
endif()

add_cpplint_target(${TARGET_NAME}_cpplint FOR_TARGETS ${TARGET_NAME})

install(TARGETS ${TARGET_NAME}
    RUNTIME DESTINATION tests
    COMPONENT tests
    EXCLUDE_FROM_ALL)

# OpenVINO 2.0 and Legacy C API test sample
set(TARGET_NAME "ov_capi_test")

file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/ov_*.cpp)
file(GLOB HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

add_executable(${TARGET_NAME} ${SOURCES} ${HEADERS})
target_link_libraries(${TARGET_NAME} PRIVATE openvino_c openvino::util
    commonTestUtils gtest_main)

target_include_directories(${TARGET_NAME} PUBLIC
    $<BUILD_INTERFACE:${OPENVINO_API_SOURCE_DIR}/include>)

target_compile_definitions(${TARGET_NAME}
    PRIVATE
    DATA_PATH=\"${DATA_PATH}\"
    MODELS_PATH=\"${MODELS_PATH}\")

if(ENABLE_AUTO OR ENABLE_MULTI)
    add_dependencies(${TARGET_NAME} openvino_auto_plugin)
endif()

if(ENABLE_AUTO_BATCH)
    add_dependencies(${TARGET_NAME} openvino_auto_batch_plugin)
endif()

if(ENABLE_INTEL_CPU)
    add_dependencies(${TARGET_NAME} openvino_intel_cpu_plugin)
endif()

if(ENABLE_INTEL_GPU)
    add_dependencies(${TARGET_NAME} openvino_intel_gpu_plugin)
endif()

add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})

install(TARGETS ${TARGET_NAME}
    RUNTIME DESTINATION tests
    COMPONENT tests
    EXCLUDE_FROM_ALL)