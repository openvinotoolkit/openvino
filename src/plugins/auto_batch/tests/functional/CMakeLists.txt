# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_auto_batch_func_tests)

set(SHARED_HEADERS_DIR "${OpenVINO_SOURCE_DIR}/src/tests/functional/plugin/shared/include")
addIeTargetTest(
        NAME
            ${TARGET_NAME}
        ROOT
            ${CMAKE_CURRENT_SOURCE_DIR}
        INCLUDES
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${SHARED_HEADERS_DIR}
        LINK_LIBRARIES
            funcSharedTests
        DEPENDENCIES
            openvino_auto_batch_plugin
        ADD_CPPLINT
        LABELS
            Auto_Batch
)

if (ENABLE_INTEL_GPU)
    find_package(OpenCL QUIET)
    if(OpenCL_FOUND)
        target_compile_definitions(${TARGET_NAME} PRIVATE HAVE_INTEL_GPU_PLUGIN)
    endif()
endif()

if (ENABLE_TEMPLATE)
    target_compile_definitions(${TARGET_NAME} PRIVATE HAVE_TEMPLATE_PLUGIN)
endif()

set_ie_threading_interface_for(${TARGET_NAME})
