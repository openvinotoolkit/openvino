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

set(DEPENDENCIES mock_engine)

if (NGRAPH_ONNX_IMPORT_ENABLE)
    set(DEPENDENCIES ${DEPENDENCIES} template_extension)
else()
    set(EXCLUDED_SOURCE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/extension)
endif()


addIeTargetTest(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        EXCLUDED_SOURCE_DIRS ${EXCLUDED_SOURCE_DIRS}
        LINK_LIBRARIES
            unitTestUtils
            inference_engine_transformations
            ${OpenCV_LIBRARIES}
        ADD_CPPLINT
        DEPENDENCIES ${DEPENDENCIES}
        LABELS
            IE
)
