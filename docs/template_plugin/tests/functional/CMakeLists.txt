# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# [cmake:functional_tests]
set(TARGET_NAME ov_template_func_tests)

ov_add_test_target(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDENCIES
            openvino_template_plugin
        LINK_LIBRARIES
            openvino::funcSharedTests
        INCLUDES
            "${TEMPLATE_PLUGIN_SOURCE_DIR}/include"
            "${CMAKE_CURRENT_SOURCE_DIR}/op_reference"
        # ADD_CPPLINT
        LABELS
            TEMPLATE
)

if(ENABLE_HETERO)
    add_dependencies(${TARGET_NAME} openvino_hetero_plugin)
endif()

find_package(OpenCV QUIET COMPONENTS core imgproc)

if(OpenCV_FOUND)
    message("-- Reference preprocessing: OpenCV tests are enabled")
    target_compile_definitions(${TARGET_NAME} PRIVATE OPENCV_TEMPLATE_TESTS)
    target_link_libraries(${TARGET_NAME} PRIVATE opencv_imgproc opencv_core)
else()
    message("-- Reference preprocessing: OpenCV tests are disabled")
endif()

# [cmake:functional_tests]
