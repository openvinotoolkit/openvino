#
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "npu_tools_utils")

#
# Define the target
#

ov_add_target(ADD_CPPLINT
              TYPE STATIC
              NAME ${TARGET_NAME}
              ROOT ${CMAKE_CURRENT_SOURCE_DIR}
              INCLUDES
                  PUBLIC
                      "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
              LINK_LIBRARIES
                  PRIVATE
                      openvino::runtime)

set_target_properties(${TARGET_NAME} PROPERTIES
                          FOLDER ${CMAKE_CURRENT_SOURCE_DIR}
                          CXX_STANDARD 17)

if (CMAKE_COMPILER_IS_GNUCXX)
    target_compile_options(${TARGET_NAME} PRIVATE -Wall)
endif()
