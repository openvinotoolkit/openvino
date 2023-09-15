# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME func_test_utils)

addIeTarget(
    NAME ${TARGET_NAME}
    TYPE STATIC
    ROOT ${CMAKE_CURRENT_SOURCE_DIR}
    ADD_CLANG_FORMAT
    DEVELOPER_PACKAGE
        tests
    INCLUDES
        PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
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
    INCLUDES
        PUBLIC
            $<TARGET_PROPERTY:openvino::runtime::dev,INTERFACE_INCLUDE_DIRECTORIES>
)

install(PROGRAMS layer_tests_summary/run_parallel.py DESTINATION tests/functional_test_utils COMPONENT tests EXCLUDE_FROM_ALL)
install(FILES layer_tests_summary/requirements.txt DESTINATION tests/functional_test_utils COMPONENT tests EXCLUDE_FROM_ALL)
install(DIRECTORY layer_tests_summary/utils DESTINATION tests/functional_test_utils COMPONENT tests EXCLUDE_FROM_ALL)

ov_build_target_faster(${TARGET_NAME}
    PCH PRIVATE "src/precomp.hpp"
)
