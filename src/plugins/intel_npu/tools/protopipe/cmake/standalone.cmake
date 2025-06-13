#
# Copyright (C) 2024 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if("${CMAKE_BUILD_TYPE}" STREQUAL "")
  set(CMAKE_BUILD_TYPE "Release")
endif()

find_package(OpenVINO REQUIRED COMPONENTS Runtime)
find_package(Threads REQUIRED)
find_package(OpenCV 4.9.0 REQUIRED COMPONENTS gapi)

find_package(yaml-cpp QUIET)
find_package(gflags QUIET)

if (NOT yaml-cpp_FOUND)
    set(YAML_CPP_SOURCES_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/yaml-cpp")
    message(STATUS "yaml-cpp package was not found. Trying to find source package in ${YAML_CPP_SOURCES_PATH}.")
    if(EXISTS ${YAML_CPP_SOURCES_PATH})
        message(STATUS "yaml-cpp source package found. yaml-cpp will be built from sources.")
        add_subdirectory(${YAML_CPP_SOURCES_PATH} yaml-cpp EXCLUDE_FROM_ALL)
    else()
        message(FATAL_ERROR "yaml-cpp package and sources were not found. CMake will exit." )
    endif()
endif()

if (NOT gflags_FOUND)
    get_filename_component(OpenVINO_PACKAGE_DIR "${OpenVINO_DIR}/../.." REALPATH)
    set(GFLAGS_SOURCES_PATH "${OpenVINO_PACKAGE_DIR}/samples/cpp/thirdparty/gflags")
    message(STATUS "gflags package was not found. Trying to find source package in ${GFLAGS_SOURCES_PATH}.")
    if(EXISTS ${GFLAGS_SOURCES_PATH})
        message(STATUS "gflags source package found. gflags will be built from sources.")
        add_subdirectory(${GFLAGS_SOURCES_PATH} gflags EXCLUDE_FROM_ALL)
    else()
        message(FATAL_ERROR "gflags was not found. CMake will exit." )
    endif()
endif()

set(DEPENDENCIES
        Threads::Threads
        gflags
        yaml-cpp
        openvino::runtime
        opencv_gapi
)

if (WIN32)
    list(APPEND DEPENDENCIES "winmm.lib")
endif()

file(GLOB_RECURSE SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
list(APPEND SOURCES main.cpp)

add_executable(${TARGET_NAME} ${SOURCES})
target_link_libraries(${TARGET_NAME} PRIVATE ${DEPENDENCIES})
target_include_directories(${TARGET_NAME} PUBLIC "${PROJECT_SOURCE_DIR}/src/")

install(TARGETS ${TARGET_NAME}
        DESTINATION "tools/${TARGET_NAME}"
        COMPONENT npu_tools)
