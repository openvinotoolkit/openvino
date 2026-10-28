#
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME app_npu_compilation_target)

ov_add_target(TYPE EXECUTABLE
              NAME ${TARGET_NAME}
              ROOT ${CMAKE_CURRENT_SOURCE_DIR}
              LINK_LIBRARIES
                  PRIVATE
                      openvino::runtime)

# npu_private_properties.hpp is header-only; avoid linking the static npu_al library
target_include_directories(${TARGET_NAME} PRIVATE
                           $<TARGET_PROPERTY:openvino_npu_al,INTERFACE_INCLUDE_DIRECTORIES>)

set_target_properties(${TARGET_NAME} PROPERTIES
                      FOLDER intel_npu_tools
                      CXX_STANDARD 17)

install(TARGETS ${TARGET_NAME}
        RUNTIME DESTINATION "tools/${TARGET_NAME}"
        COMPONENT ${NPU_INTERNAL_COMPONENT}
        ${OV_CPACK_COMP_NPU_INTERNAL_EXCLUDE_ALL})
