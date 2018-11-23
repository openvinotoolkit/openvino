# Copyright (C) 2018 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
####################################
## All next project will use C++11
set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
####################################


set (gtest_force_shared_crt ON CACHE BOOL "disable static CRT for google test")

#detecting regex support
if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.9)
    add_definitions(-DUSE_BOOST_RE)
    set(USE_BOOST_RE ON)
else()
    set(USE_BOOST_RE OFF)
endif()

add_subdirectory(mock_engine)

add_subdirectory(libs/gtest)

include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR}
        ${gmock_SOURCE_DIR}/include
        ${gmock_SOURCE_DIR}
        ${IE_MAIN_SOURCE_DIR}/include
        ${IE_MAIN_SOURCE_DIR}/src
        ${IE_MAIN_SOURCE_DIR}/thirdparty/pugixml/src
        ${IE_MAIN_SOURCE_DIR}/tests/helpers
        ${IE_MAIN_SOURCE_DIR}/samples/common
        ${IE_MAIN_SOURCE_DIR}/samples/common/format_reader
        ${MKLDNN}/include)

#####################################################################################################
#                                     SETUP GOOGLE TESTS                                            #
#####################################################################################################
enable_testing()

link_directories(${LIBRARY_OUTPUT_DIRECTORY})

add_subdirectory(helpers)
add_subdirectory(unit)
