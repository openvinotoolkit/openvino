# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME bytecode_interpreter)

set(GFLAGS_USE_TARGET_NAMESPACE ON)
find_package(gflags QUIET)

ov_add_target(TYPE EXECUTABLE
              NAME ${TARGET_NAME}
              ROOT ${CMAKE_CURRENT_SOURCE_DIR}
              LINK_LIBRARIES
                  PRIVATE
                      openvino::runtime::dev
                        gflags
                        openvino::intel_npu_vm_runtime
                        openvino::npu_zero_utils)

set_target_properties(${TARGET_NAME} PROPERTIES
                      FOLDER intel_npu_tools
                      CXX_STANDARD 17)

#
# Install
#

install(TARGETS ${TARGET_NAME}
        RUNTIME DESTINATION "tools/${TARGET_NAME}"
        COMPONENT ${NPU_INTERNAL_COMPONENT})