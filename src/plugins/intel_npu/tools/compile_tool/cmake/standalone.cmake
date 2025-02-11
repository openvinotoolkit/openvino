#
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if("${CMAKE_BUILD_TYPE}" STREQUAL "")
  set(CMAKE_BUILD_TYPE "Release")
endif()

#
# OpenVINO package path should be specified via OpenVINO_DIR
#

find_package(Threads REQUIRED)
find_package(OpenVINO REQUIRED COMPONENTS Runtime)

#
# gflags is distributed in sources in OpenVINO packages so we need to build it explicitly
#

if(EXISTS "${PACKAGE_PREFIX_DIR}/samples/cpp/thirdparty/gflags")
    add_subdirectory("${PACKAGE_PREFIX_DIR}/samples/cpp/thirdparty/gflags" gflags EXCLUDE_FROM_ALL)
else()
    find_package(gflags REQUIRED)
endif()

set(DEPENDENCIES
        Threads::Threads
        gflags
        openvino::runtime
)

file(GLOB SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_executable(${TARGET_NAME} ${SOURCES})
target_link_libraries(${TARGET_NAME} ${DEPENDENCIES})

if (CMAKE_COMPILER_IS_GNUCXX)
    target_compile_options(${TARGET_NAME} PRIVATE -Wall)
endif()

install(TARGETS ${TARGET_NAME}
        DESTINATION "tools/${TARGET_NAME}"
        COMPONENT npu_tools)
