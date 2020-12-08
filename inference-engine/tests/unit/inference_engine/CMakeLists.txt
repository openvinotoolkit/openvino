# Copyright (C) 2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ieUnitTests)

# Find OpenCV components if exist
find_package(OpenCV COMPONENTS imgcodecs videoio imgproc QUIET)
if(NOT OpenCV_FOUND)
    message(WARNING "OPENCV is disabled or not found, " ${TARGET_NAME} " needs OpenCV for its build")
else()
    add_definitions(-DUSE_OPENCV)
endif()

addIeTargetTest(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        LINK_LIBRARIES
            unitTestUtils
            inference_engine_lp_transformations
            ${OpenCV_LIBRARIES}
        ADD_CPPLINT
        DEPENDENCIES
            template_extension
            mock_engine
        LABELS
            IE
)
