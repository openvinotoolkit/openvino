# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# if(NOT OV_COMPILER_IS_INTEL_LLVM)
#     return()
# endif()

set(TARGET_NAME "openvino_intel_gpu_sycl_obj")

ov_gpu_add_backend_target(
    NAME ${TARGET_NAME}
)

# add_sycl_to_target(TARGET ${TARGET_NAME})

file(GLOB LIBRARY_SOURCES_SYCL
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

set_source_files_properties(SOURCE ${LIBRARY_SOURCES_SYCL} PROPERTIES
    COMPILE_FLAGS "-fsycl"
)
