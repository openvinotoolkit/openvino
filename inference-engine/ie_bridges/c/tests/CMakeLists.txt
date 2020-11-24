# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "InferenceEngineCAPITests")

# Find OpenCV components if exist
find_package(OpenCV COMPONENTS imgcodecs videoio imgproc QUIET)
if(NOT OpenCV_FOUND)
    message(WARNING "OPENCV is disabled or not found, " ${TARGET_NAME} " is disabled")
    return()
endif()

add_executable(${TARGET_NAME} ie_c_api_test.cpp test_model_repo.hpp)

target_link_libraries(${TARGET_NAME}
        PRIVATE
        inference_engine
        inference_engine_c_api
        ${OpenCV_LIBRARIES}
        commonTestUtils
        )

target_compile_definitions(${TARGET_NAME}
        PUBLIC ${ARGV}
        DATA_PATH=\"${DATA_PATH}\"
        MODELS_PATH=\"${MODELS_PATH}\" )

add_dependencies(${TARGET_NAME} MultiDevicePlugin)

if(ENABLE_MKL_DNN)
    add_dependencies(${TARGET_NAME} MKLDNNPlugin)
endif()

if(ENABLE_CLDNN)
    add_dependencies(${TARGET_NAME} clDNNPlugin)
endif()

add_cpplint_target(${TARGET_NAME}_cpplint FOR_TARGETS ${TARGET_NAME})
