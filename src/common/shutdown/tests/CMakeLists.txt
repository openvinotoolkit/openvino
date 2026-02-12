# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME shutdown_test_lib)

# Build as MODULE library so tests can dlopen/dlclose it (like plugins)
add_library(${TARGET_NAME} MODULE test_lib.cpp)
target_link_libraries(${TARGET_NAME} PRIVATE openvino::shutdown)

set_target_properties(${TARGET_NAME} PROPERTIES
    EXPORT_NAME shutdown_test_lib
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
)

install(TARGETS ${TARGET_NAME}
        LIBRARY DESTINATION tests COMPONENT tests EXCLUDE_FROM_ALL)

set(TEST_TARGET_NAME ov_shutdown_tests)

ov_add_test_target(
    NAME ${TEST_TARGET_NAME}
    ROOT ${CMAKE_CURRENT_SOURCE_DIR}
    EXCLUDED_SOURCE_PATHS ${CMAKE_CURRENT_SOURCE_DIR}/test_lib.cpp
    DEPENDENCIES
        shutdown_test_lib
        openvino::shutdown
    LINK_LIBRARIES
        gmock
        func_test_utils
    ADD_CLANG_FORMAT
    LABELS
        OV UNIT TRANSFORMATIONS
)

set_source_files_properties(test_lib.cpp shutdown_test.cpp
        PROPERTIES INCLUDE_DIRECTORIES $<TARGET_PROPERTY:openvino::core::dev,INTERFACE_INCLUDE_DIRECTORIES>)
