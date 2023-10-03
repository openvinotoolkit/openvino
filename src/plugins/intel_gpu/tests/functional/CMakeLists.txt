# Copyright (C) 2018-2023 Intel Corporation
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

addIeTargetTest(
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
        DEPENDENCIES
            openvino_intel_gpu_plugin
        LINK_LIBRARIES
            funcSharedTests
            OpenCL::NewHeaders # should come before OpenCL::OpenCL
            OpenCL::OpenCL
        ADD_CPPLINT
        LABELS
            GPU
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

if (ENABLE_INTEL_CPU)
    set_source_files_properties(
        "${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/behavior/ov_plugin/life_time.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/behavior/ov_plugin/properties_tests.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/behavior/ov_infer_request/perf_counters.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/multi/gpu_remote_blob_tests.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/behavior/infer_request/memory_states.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/behavior/ov_executable_network/exec_net_base.cpp"
        PROPERTIES COMPILE_DEFINITIONS ENABLE_INTEL_CPU=1)
endif()
