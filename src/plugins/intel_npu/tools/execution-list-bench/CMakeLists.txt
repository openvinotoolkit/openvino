#
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME execution-list-bench)

#
# Define the target
#

ov_add_target(TYPE EXECUTABLE
              NAME ${TARGET_NAME}
              ROOT ${CMAKE_CURRENT_SOURCE_DIR}
              LINK_LIBRARIES
                  PRIVATE
                      openvino::runtime::dev
                      gflags)

# Proof of concept: the execution list interface is internal to the NPU plugin, it only uses the OpenVINO dev API
target_include_directories(${TARGET_NAME} PRIVATE
                           "${CMAKE_CURRENT_SOURCE_DIR}/../../src/common/include")

set_target_properties(${TARGET_NAME} PROPERTIES
                      FOLDER intel_npu_tools
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
