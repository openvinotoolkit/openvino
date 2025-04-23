#
# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME compile_tool)

if (NOT DEFINED PROJECT_NAME)
    if(WIN32)
        cmake_minimum_required(VERSION 3.16)
    else()
        cmake_minimum_required(VERSION 3.13)
    endif()
    project(compile_tool_standalone)
    include("cmake/standalone.cmake")
    return()
endif()

#
# Define the target
#

ov_add_target(ADD_CPPLINT
              TYPE EXECUTABLE
              NAME ${TARGET_NAME}
              ROOT ${CMAKE_CURRENT_SOURCE_DIR}
              LINK_LIBRARIES
                  PRIVATE
                      openvino::runtime
                      gflags
                      Threads::Threads
                      npu_tools_utils)

set_target_properties(${TARGET_NAME} PROPERTIES
                          FOLDER ${CMAKE_CURRENT_SOURCE_DIR}
                          CXX_STANDARD 17)

# TODO: fix warnings and remove this exception
if(CMAKE_COMPILER_IS_GNUCXX OR OV_COMPILER_IS_CLANG)
    ov_add_compiler_flags(-Wno-missing-declarations)
endif()

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
