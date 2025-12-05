#
# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME npu_metadata_parser)

if (NOT DEFINED PROJECT_NAME)
    if(WIN32)
        cmake_minimum_required(VERSION 3.16)
    else()
        cmake_minimum_required(VERSION 3.13)
    endif()
    project(npu_metadata_parser_standalone)
    include("cmake/standalone.cmake")
    return()
endif()

#
# Define the target
#

ov_add_target(TYPE EXECUTABLE
              NAME ${TARGET_NAME}
              ROOT ${CMAKE_CURRENT_SOURCE_DIR}
              LINK_LIBRARIES
                  PRIVATE
                      openvino::core::dev
                      openvino::runtime
                      gflags
                      openvino_npu_logger_utils
              INCLUDES "${NPU_PLUGIN_SOURCE_DIR}/src/plugin/include"
              OBJECT_FILES "${NPU_PLUGIN_SOURCE_DIR}/src/plugin/src/metadata.cpp")

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
