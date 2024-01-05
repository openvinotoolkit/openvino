# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_thread_local_test)

#target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

#add_subdirectory(${OpenVINO_SOURCE_DIR}/thirdparty/gtest
#                 ${CMAKE_CURRENT_BINARY_DIR}/gtest_build
#                EXCLUDE_FROM_ALL)

# add ov_thread_local_test tool
#ov_add_test_target(
#        NAME ${TARGET_NAME}
#        TYPE EXECUTABLE
#        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
#        INCLUDES
#            ${CMAKE_CURRENT_SOURCE_DIR}/
#        LINK_LIBRARIES
#            gtest
#        ADD_CPPLINT
#)

file (GLOB_RECURSE SRC *.cpp)
file (GLOB_RECURSE HDR *.h)

set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/test")

# Create library file from sources.
add_executable(${TARGET_NAME} ${HDR} ${SRC})

target_link_libraries(${TARGET_NAME}
    PUBLIC
        gtest
        pugixml
        gflags
    PRIVATE
        gtest_main)


