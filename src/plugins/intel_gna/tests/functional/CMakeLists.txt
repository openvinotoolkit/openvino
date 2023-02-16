# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    ie_add_compiler_flags(-Wno-literal-conversion)
endif()

set(TARGET_NAME ov_gna_func_tests)

addIeTargetTest(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        INCLUDES
            ${CMAKE_CURRENT_SOURCE_DIR}
            $<TARGET_PROPERTY:openvino_intel_gna_plugin,INCLUDE_DIRECTORIES>
            $<TARGET_PROPERTY:inference_engine_legacy,INTERFACE_INCLUDE_DIRECTORIES>
            $<TARGET_PROPERTY:libGNA::API,INTERFACE_INCLUDE_DIRECTORIES>
        DEPENDENCIES
            openvino_intel_gna_plugin
        LINK_LIBRARIES
            funcSharedTests
            openvino_intel_gna_plugin_test_static
            inference_engine_legacy_s
            libGNA
        ADD_CLANG_FORMAT
        ADD_CPPLINT
        LABELS
            GNA
)
