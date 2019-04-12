#
# Copyright (C) 2018-2019 Intel Corporation.
#
# This software and the related documents are Intel copyrighted materials,
# and your use of them is governed by the express license under which they
# were provided to you (End User License Agreement for the Intel(R) Software
# Development Products (Version May 2017)). Unless the License provides
# otherwise, you may not use, modify, copy, publish, distribute, disclose or
# transmit this software or the related documents without Intel's prior
# written permission.
#
# This software and the related documents are provided as is, with no
# express or implied warranties, other than those that are expressly
# stated in the License.
#

set (TARGET_NAME "test_validation_app")

# Find OpenCV components if exist
find_package(OpenCV COMPONENTS imgcodecs)
if(NOT(OpenCV_FOUND))
    message(WARNING "No suitable OpenCV version detected, " ${TARGET_NAME} " skipped")
    return()
endif()

set(VALIDATION_APP_SOURCE "${IE_MAIN_SOURCE_DIR}/samples/validation_app")

file (GLOB MAIN_SRC
        ${VALIDATION_APP_SOURCE}/*.cpp
        ${VALIDATION_APP_SOURCE}/pugixml/*.cpp
        )

file (GLOB MAIN_HEADERS
        ${VALIDATION_APP_SOURCE}/*.hpp
        ${VALIDATION_APP_SOURCE}/pugixml/*.hpp
        )

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj
source_group("src" FILES ${MAIN_SRC})
source_group("include" FILES ${MAIN_HEADERS})

if (WIN32)
    if(NOT "${CMAKE_SIZEOF_VOID_P}" EQUAL "8")
        message(FATAL_ERROR "Only 64-bit supported on Windows")
    endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_SCL_SECURE_NO_WARNINGS -DNOMINMAX")
endif()

# Properties->C/C++->General->Additional Include Directories
include_directories (${VALIDATION_APP_SOURCE}/../classification_sample/core
        ${VALIDATION_APP_SOURCE}/../common
        ${VALIDATION_APP_SOURCE}/../common/os/windows
        ${VALIDATION_APP_SOURCE}/../../include)

# Create library file from sources.

list(REMOVE_ITEM MAIN_SRC ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)

add_library(${TARGET_NAME} STATIC ${MAIN_SRC} ${MAIN_HEADERS})
set_target_properties(${TARGET_NAME} PROPERTIES "COMPILE_PDB_NAME" ${TARGET_NAME})
target_link_libraries(${TARGET_NAME} gflags ie_cpu_extension ${OpenCV_LIBRARIES})