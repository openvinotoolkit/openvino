# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME func_test_utils)

ov_add_target(
    NAME ${TARGET_NAME}
    TYPE STATIC
    ROOT ${CMAKE_CURRENT_SOURCE_DIR}
    ADD_CLANG_FORMAT
    DEVELOPER_PACKAGE
        tests
    INCLUDES
        PUBLIC
            "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
            "$<INSTALL_INTERFACE:developer_package/include/func_test_utils>"
    ADDITIONAL_SOURCE_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    LINK_LIBRARIES
        PUBLIC
            common_test_utils
            openvino::runtime
            openvino::runtime::dev
        PRIVATE
            ov_models
            openvino::pugixml
)

ov_build_target_faster(${TARGET_NAME}
    PCH PRIVATE "src/precomp.hpp"
)

# installation

install(DIRECTORY layer_tests_summary DESTINATION tests/functional_test_utils
        COMPONENT tests EXCLUDE_FROM_ALL)

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
        DESTINATION developer_package/include/func_test_utils
        COMPONENT developer_package EXCLUDE_FROM_ALL)
