# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "XLinkTests")
set(CMAKE_CXX_STANDARD 11)

include(${XLINK_DIR}/XLink.cmake)

file(GLOB_RECURSE XLINK_TESTS_SOURCES "*.cpp")

add_executable(${TARGET_NAME} ${XLINK_TESTS_SOURCES})

target_include_directories(${TARGET_NAME}
        PRIVATE
        helpers
        cases
        ${XLINK_INCLUDE}
        ${XLINK_PLATFORM_INCLUDE})

target_compile_definitions(${TARGET_NAME}
        PRIVATE
        __PC__)

target_link_libraries(${TARGET_NAME}
        PRIVATE
        XLink gtest gtest_main)

set_target_properties(${TARGET_NAME} PROPERTIES
        POSITION_INDEPENDENT_CODE TRUE
        COMPILE_PDB_NAME ${TARGET_NAME})

add_dependencies(${TARGET_NAME} vpu_copy_firmware)

install(TARGETS ${TARGET_NAME}
        RUNTIME DESTINATION tests
        COMPONENT tests
        EXCLUDE_FROM_ALL)
