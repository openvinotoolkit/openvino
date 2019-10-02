# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "XLinkTests")
set(CMAKE_CXX_STANDARD 11)

file(GLOB_RECURSE XLINK_TESTS_SOURCES "*.cpp")

add_executable(${TARGET_NAME} ${XLINK_TESTS_SOURCES})

target_include_directories(${TARGET_NAME}
        PRIVATE
        helpers
        cases
        ${IE_MAIN_SOURCE_DIR}/tests/libs/gtest/googletest/include
        ${IE_MAIN_SOURCE_DIR}/tests/libs/gtest/googletest/
        ${XLINK_DIR}/shared
        ${XLINK_DIR}/pc)

if(WIN32)
    target_include_directories(${TARGET_NAME}
            PRIVATE
            "../../USB_WIN")
else()
    target_include_directories(${TARGET_NAME}
            PRIVATE
            "${LIBUSB_INCLUDE_DIR}")
endif()

target_include_directories(${TARGET_NAME}
        PRIVATE
        "../../WinPthread")


target_compile_definitions(${TARGET_NAME}
        PRIVATE
        __PC__)

target_link_libraries(${TARGET_NAME}
        PRIVATE
        XLink gtest gtest_main)
        
set_target_properties(${TARGET_NAME} PROPERTIES
        POSITION_INDEPENDENT_CODE TRUE
        COMPILE_PDB_NAME ${TARGET_NAME})
