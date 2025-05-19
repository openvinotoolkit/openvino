#
# Copyright (C) 2023-2024 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

set(TARGET_NAME protopipe)

if (NOT DEFINED PROJECT_NAME)
    set(STANDALONE true)
endif()

# Enable CMP0048 policy
if(POLICY CMP0048)
    cmake_policy(SET CMP0048 NEW)
endif()

project(protopipe VERSION 1.0.0)

execute_process(
  COMMAND git rev-parse --short HEAD
  OUTPUT_VARIABLE GIT_SHA
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/version.hpp.in
  ${CMAKE_CURRENT_BINARY_DIR}/version/version.hpp
  @ONLY
)

if (STANDALONE)
    if(WIN32)
        cmake_minimum_required(VERSION 3.16)
    else()
        cmake_minimum_required(VERSION 3.13)
    endif()
    include("cmake/standalone.cmake")
    return()
endif()

#
# Dependencies
#

find_package(OpenCV QUIET COMPONENTS gapi)
if(OpenCV_VERSION VERSION_LESS 4.9)
    message(STATUS "NPU ${TARGET_NAME} tool is disabled due to missing dependencies: gapi from OpenCV >= 4.9.")
    return()
endif()

set(DEPENDENCIES
        Threads::Threads
        gflags
        yaml-cpp
        openvino::runtime
        opencv_gapi
)

if (WIN32)
    # WA: add_tool_target expects to have all dependencies as cmake targets.
    add_library(winmm INTERFACE)
    target_link_libraries(winmm INTERFACE "winmm.lib")
    list(APPEND DEPENDENCIES winmm)
endif()

#
# Define the target
#

set(PROTOPIPE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

ov_add_target(ADD_CPPLINT
              TYPE EXECUTABLE
              NAME ${TARGET_NAME}
              ROOT ${CMAKE_CURRENT_SOURCE_DIR}
              ADDITIONAL_SOURCE_DIRS ${PROTOPIPE_SOURCE_DIR}
              INCLUDES ${PROTOPIPE_SOURCE_DIR} "${CMAKE_CURRENT_BINARY_DIR}/version"
              LINK_LIBRARIES PRIVATE ${DEPENDENCIES})

set_target_properties(${TARGET_NAME} PROPERTIES
                          FOLDER ${CMAKE_CURRENT_SOURCE_DIR}
                          CXX_STANDARD 17)

#
# Install
#

install(TARGETS ${TARGET_NAME}
        RUNTIME DESTINATION "tools/${TARGET_NAME}"
        COMPONENT ${NPU_INTERNAL_COMPONENT}
        ${OV_CPACK_COMP_NPU_INTERNAL_EXCLUDE_ALL})

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/README.md"
            DESTINATION "tools/${TARGET_NAME}"
            COMPONENT ${NPU_INTERNAL_COMPONENT}
            ${OV_CPACK_COMP_NPU_INTERNAL_EXCLUDE_ALL})
endif()
