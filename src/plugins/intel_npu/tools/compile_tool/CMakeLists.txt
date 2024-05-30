#
# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME compile_tool)

if (NOT DEFINED PROJECT_NAME)
    cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
    project(compile_tool_standalone)
    include("cmake/standalone.cmake")
    return()
endif()

set(LINK_LIBRARIES
    openvino::runtime
    gflags
    Threads::Threads
)

#
# Define the target
#

ov_add_target(ADD_CPPLINT
              TYPE EXECUTABLE
              NAME ${TARGET_NAME}
              ROOT ${CMAKE_CURRENT_SOURCE_DIR}
              LINK_LIBRARIES PRIVATE ${LINK_LIBRARIES})

set_target_properties(${TARGET_NAME} PROPERTIES
                          FOLDER ${CMAKE_CURRENT_SOURCE_DIR}
                          CXX_STANDARD 17)

if (CMAKE_COMPILER_IS_GNUCXX)
    target_compile_options(${TARGET_NAME} PRIVATE -Wall)
endif()

#
# Install
#

install(TARGETS ${TARGET_NAME}
        RUNTIME DESTINATION "tools/${TARGET_NAME}"
        COMPONENT ${NPU_INTERNAL_COMPONENT}
        EXCLUDE_FROM_ALL)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/README.md"
            DESTINATION "tools/${TARGET_NAME}"
            COMPONENT ${NPU_INTERNAL_COMPONENT}
            EXCLUDE_FROM_ALL)
endif()