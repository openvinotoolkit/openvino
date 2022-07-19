# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "InferenceEngineCAPITests")

# Find OpenCV components if exist
find_package(OpenCV COMPONENTS core imgproc imgcodecs QUIET)

add_executable(${TARGET_NAME} ie_c_api_test.cpp test_model_repo.hpp)

if(OpenCV_FOUND)
    target_compile_definitions(${TARGET_NAME} PRIVATE BUILD_OPENCV_DEPENDENT_TESTS)
endif()

target_link_libraries(${TARGET_NAME} PRIVATE openvino_c ${OpenCV_LIBRARIES}
                                             commonTestUtils gtest_main)

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
