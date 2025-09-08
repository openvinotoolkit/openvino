#
# Copyright (C) 2024 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if("${CMAKE_BUILD_TYPE}" STREQUAL "")
  set(CMAKE_BUILD_TYPE "Release")
endif()

#
# OpenVINO and OpenCV package paths should be specified via OpenVINO_DIR and OpenCV_DIR
# TBB is distributed with OpenVINO packages however it is not exported from OpenVINO cmake config
# Therefore we need to find TBB explicitly. TBB_DIR is not requered as OpenVINO setupvars script sets its location
#

find_package(Threads REQUIRED)
find_package(OpenVINO REQUIRED COMPONENTS Runtime)
find_package(TBB QUIET)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../common" common EXCLUDE_FROM_ALL)

#
# gflags is distributed in sources in OpenVINO packages so we need to build it explicitly
#

get_filename_component(OpenVINO_PACKAGE_DIR "${OpenVINO_DIR}/../.." REALPATH)
if(EXISTS "${OpenVINO_PACKAGE_DIR}/samples/cpp/thirdparty/gflags")
    add_subdirectory("${OpenVINO_PACKAGE_DIR}/samples/cpp/thirdparty/gflags" gflags EXCLUDE_FROM_ALL)
else()
    find_package(gflags REQUIRED)
endif()

set(DEPENDENCIES
        Threads::Threads
        gflags
        openvino::runtime
        opencv_core
        opencv_imgproc
        opencv_imgcodecs
        npu_tools_utils
)

if (TBB_FOUND)
    list(APPEND DEPENDENCIES TBB::tbb)
else()
    message(WARNING
        "TBB not found. We will try to build SIT without TBB. "
        "If OpenVINO was built with TBB, you'll likely get a linking error. "
        "Make sure you have called setupvars or have specified TBB_DIR.")
endif()

file(GLOB SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_executable(${TARGET_NAME} ${SOURCES})
target_link_libraries(${TARGET_NAME} PRIVATE ${DEPENDENCIES})
if (CMAKE_COMPILER_IS_GNUCXX)
    target_compile_options(${TARGET_NAME} PRIVATE -Wall)
endif()

install(TARGETS ${TARGET_NAME}
        DESTINATION "tools/${TARGET_NAME}"
        COMPONENT npu_tools)
