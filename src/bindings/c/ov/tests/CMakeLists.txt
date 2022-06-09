# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "OpenVinoCAPITests")

# Find OpenCV components if exist
find_package(OpenCV COMPONENTS core imgproc imgcodecs QUIET)
if(NOT OpenCV_FOUND)
    message(WARNING "OPENCV is disabled or not found, ${TARGET_NAME} is disabled")
    return()
endif()

add_executable(${TARGET_NAME} ov_c_api_test.cpp test_model_repo.hpp)

target_link_libraries(${TARGET_NAME} PRIVATE openvino_ov_c ${OpenCV_LIBRARIES}
                                             commonTestUtils gtest_main)

target_include_directories(${TARGET_NAME} PUBLIC
    $<BUILD_INTERFACE:${InferenceEngine_C_API_SOURCE_DIR}/include>)

target_compile_definitions(${TARGET_NAME}
        PRIVATE
        $<$<BOOL:${ENABLE_GAPI_PREPROCESSING}>:ENABLE_GAPI_PREPROCESSING>
        DATA_PATH=\"${DATA_PATH}\"
        MODELS_PATH=\"${MODELS_PATH}\" )

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
