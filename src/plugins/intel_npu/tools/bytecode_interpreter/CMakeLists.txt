#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME bytecode_interpreter)

set(GFLAGS_USE_TARGET_NAMESPACE ON)
find_package(gflags QUIET)

add_tool_target(
    NAME ${TARGET_NAME}
    ROOT ${CMAKE_CURRENT_SOURCE_DIR}
    LINK_LIBRARIES
        openvino::runtime::dev
        openvino::gflags_nothreads_static
        openvino_intel_npu_vm_runtime
)
