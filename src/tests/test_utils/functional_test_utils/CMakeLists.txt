# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME func_test_utils)

ov_add_target(
    NAME ${TARGET_NAME}
    TYPE STATIC
    ROOT ${CMAKE_CURRENT_SOURCE_DIR}
    ADD_CLANG_FORMAT
    INCLUDES
        PUBLIC
            "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    ADDITIONAL_SOURCE_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    LINK_LIBRARIES
        PUBLIC
            common_test_utils
            openvino::runtime
            openvino::runtime::dev
        PRIVATE
            openvino::pugixml
)

ov_build_target_faster(${TARGET_NAME}
    PCH PRIVATE "src/precomp.hpp"
)

# install & export

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/layer_tests_summary"
        DESTINATION tests/functional_test_utils
        COMPONENT tests EXCLUDE_FROM_ALL)

ov_developer_package_export_targets(TARGET ${TARGET_NAME}
                                    INSTALL_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/include/")
