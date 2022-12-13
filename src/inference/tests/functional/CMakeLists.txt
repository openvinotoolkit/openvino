# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_inference_functional_tests)

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
    ie_add_compiler_flags(-Wno-suggest-override)
endif()

set(DEPENDENCIES
    mock_engine
    template_extension
    openvino_template_extension
)

set(COMPILE_DEFINITIONS "")

if(ENABLE_OV_IR_FRONTEND)
    list(APPEND DEPENDENCIES openvino_ir_frontend)
    list(APPEND COMPILE_DEFINITIONS ENABLE_OV_IR_FRONTEND)
endif()

if(ENABLE_HETERO)
    list(APPEND DEPENDENCIES openvino_hetero_plugin)
    list(APPEND COMPILE_DEFINITIONS ENABLE_HETERO)
endif()

if(ENABLE_AUTO AND ENABLE_MULTI)
    list(APPEND DEPENDENCIES openvino_auto_plugin)
    list(APPEND COMPILE_DEFINITIONS ENABLE_AUTO)
endif()

if(ENABLE_AUTO_BATCH)
    list(APPEND DEPENDENCIES openvino_auto_batch_plugin)
    list(APPEND COMPILE_DEFINITIONS ENABLE_AUTO_BATCH)
endif()


ov_add_test_target(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDENCIES
            ${DEPENDENCIES}
        LINK_LIBRARIES
            gmock
            funcTestUtils
        INCLUDES
            $<TARGET_PROPERTY:inference_engine_obj,SOURCE_DIR>/src
        ADD_CLANG_FORMAT
        LABELS
            OV
)

add_compile_definitions(${TARGET_NAME} ${COMPILE_DEFINITIONS})

set(legacy_tests "${CMAKE_CURRENT_SOURCE_DIR}/cnn_ngraph_impl_tests.cpp") #CVS-55375
set_source_files_properties(${legacy_tests} PROPERTIES INCLUDE_DIRECTORIES
    $<TARGET_PROPERTY:inference_engine_legacy,INTERFACE_INCLUDE_DIRECTORIES>)
