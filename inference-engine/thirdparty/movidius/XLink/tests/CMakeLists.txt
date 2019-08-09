# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "XLinkTests")
set(CMAKE_CXX_STANDARD 11)

add_executable(${TARGET_NAME} XLink_tests.cpp)

target_include_directories(${TARGET_NAME}
        PRIVATE
            ${IE_MAIN_SOURCE_DIR}/tests/libs/gtest/googletest/include
            ${IE_MAIN_SOURCE_DIR}/tests/libs/gtest/googletest/
            ../shared
            ../pc)

target_link_libraries(${TARGET_NAME}
        PRIVATE
            XLink gtest gtest_main)

set_target_properties(${TARGET_NAME} PROPERTIES
        POSITION_INDEPENDENT_CODE TRUE
        COMPILE_PDB_NAME ${TARGET_NAME})
