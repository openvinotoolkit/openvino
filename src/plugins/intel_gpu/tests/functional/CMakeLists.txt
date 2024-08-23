# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_gpu_func_tests)

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # C4267, 4244 issues from oneDNN headers conversion from 'XXX' to 'YYY', possible loss of data
    ov_add_compiler_flags(/wd4267)
    ov_add_compiler_flags(/wd4244)
    # 'initializing': truncation from 'XXX' to 'YYY'
    ov_add_compiler_flags(/wd4305)
endif()

list(APPEND DEFINES TEST_CUSTOM_OP_CONFIG_PATH="${CMAKE_CURRENT_SOURCE_DIR}/custom_op/custom_op.xml")

ov_add_test_target(
        NAME
            ${TARGET_NAME}
        ROOT
            ${CMAKE_CURRENT_SOURCE_DIR}
        ADDITIONAL_SOURCE_DIRS
            ${TEST_COMMON_SOURCE_DIR}
        INCLUDES
            ${CMAKE_CURRENT_SOURCE_DIR}
            $<TARGET_PROPERTY:openvino_intel_gpu_plugin,SOURCE_DIR>/include/
            ${TEST_COMMON_INCLUDE_DIR}
        DEFINES
            ${DEFINES}
        DEPENDENCIES
            openvino_intel_gpu_plugin
        LINK_LIBRARIES
            openvino::reference
            funcSharedTests
            OpenCL::NewHeaders # should come before OpenCL::OpenCL
            OpenCL::OpenCL
        ADD_CPPLINT
        LABELS
            OV GPU
)

if(ENABLE_PROXY)
    target_compile_definitions(${TARGET_NAME} PUBLIC PROXY_PLUGIN_ENABLED)
endif()

# try to find VA libraries
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_search_module(libva QUIET IMPORTED_TARGET libva)
endif()

if(libva_FOUND)
    target_compile_definitions(${TARGET_NAME} PRIVATE ENABLE_LIBVA)
    target_link_libraries(${TARGET_NAME} PRIVATE PkgConfig::libva)
endif()

if(WIN32)
    target_compile_definitions(${TARGET_NAME} PRIVATE ENABLE_DX11)
    target_link_libraries(${TARGET_NAME} PRIVATE d3d11 dxgi)
endif()
