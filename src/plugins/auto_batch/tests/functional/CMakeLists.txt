# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_auto_batch_func_tests)

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    ie_add_compiler_flags(/wd4244)
    ie_add_compiler_flags(/wd4267)
    ie_add_compiler_flags(/wd4305)
    ie_add_compiler_flags(/wd4018)
    ie_add_compiler_flags(/wd4050)
    ie_add_compiler_flags(/wd4250)
    ie_add_compiler_flags(/wd4334)
    ie_add_compiler_flags(/wd4661)
    ie_add_compiler_flags(/wd4273)
    ie_add_compiler_flags(/wd4309)
    ie_add_compiler_flags(/wd4804)
endif()
if(NOT MSVC)
    ie_add_compiler_flags(-Wno-missing-declarations)
    ie_add_compiler_flags(-Wno-sign-compare)
    ie_add_compiler_flags(-Wno-odr)
    ie_add_compiler_flags(-Wno-all)
endif()

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
    target_compile_definitions(${TARGET_NAME} PRIVATE HAVE_INTEL_GPU_PLUGIN)
endif()
if (ENABLE_TEMPLATE)
    target_compile_definitions(${TARGET_NAME} PRIVATE HAVE_TEMPLATE_PLUGIN)
endif()

set_ie_threading_interface_for(${TARGET_NAME})
