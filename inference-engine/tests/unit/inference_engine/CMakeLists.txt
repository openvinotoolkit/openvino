# Copyright (C) 2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ieUnitTests)

# Find OpenCV components if exist
find_package(OpenCV COMPONENTS imgcodecs videoio imgproc QUIET)
if(NOT OpenCV_FOUND)
    message(ERROR "OPENCV is disabled or not found, " ${TARGET_NAME} " needs OpenCV for its build")
else()
    add_definitions(-DUSE_OPENCV)
endif()

addIeTargetTest(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        LINK_LIBRARIES
            unitTestUtils
            ${OpenCV_LIBRARIES}
        ADD_CPPLINT
        DEPENDENCIES
            mock_engine
        LABELS
            IE
)

target_include_directories(${TARGET_NAME} PRIVATE
    $<TARGET_PROPERTY:inference_engine_ir_v7_reader,INTERFACE_INCLUDE_DIRECTORIES>)
