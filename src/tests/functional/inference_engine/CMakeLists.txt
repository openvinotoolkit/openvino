# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ieFuncTests)

set(INCLUDES $<TARGET_PROPERTY:inference_engine_obj,SOURCE_DIR>/src)

set(LINK_LIBRARIES
    gmock
    funcTestUtils
    ngraphFunctions
    lptNgraphFunctions
    openvino::itt
    openvino::util
    openvino::conditional_compilation
    sharedTestClasses
    offline_transformations
    pugixml::static
    openvino::runtime::dev
)

set(DEPENDENCIES
    mock_engine
    template_extension
    openvino_template_extension
    lptNgraphFunctions
    sharedTestClasses
    test_model_zoo
)

if(ENABLE_OV_IR_FRONTEND)
    list(APPEND DEPENDENCIES openvino_ir_frontend)
endif()

if(ENABLE_IR_V7_READER)
    list(APPEND DEPENDENCIES inference_engine_ir_v7_reader)
endif()

if(ENABLE_HETERO)
    list(APPEND DEPENDENCIES openvino_hetero_plugin)
endif()

if(ENABLE_AUTO OR ENABLE_MULTI)
    list(APPEND DEPENDENCIES openvino_auto_plugin)
endif()

if(ENABLE_AUTO_BATCH)
    list(APPEND DEPENDENCIES openvino_auto_batch_plugin)
endif()

if (NOT ENABLE_OV_ONNX_FRONTEND)
    list(APPEND EXCLUDED_SOURCE_PATHS "${CMAKE_CURRENT_SOURCE_DIR}/onnx_reader")
endif()

if (NOT ENABLE_OV_PADDLE_FRONTEND)
    list(APPEND EXCLUDED_SOURCE_PATHS "${CMAKE_CURRENT_SOURCE_DIR}/paddle_reader")
endif()

addIeTargetTest(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        INCLUDES ${INCLUDES}
        EXCLUDED_SOURCE_PATHS ${EXCLUDED_SOURCE_PATHS}
        LINK_LIBRARIES ${LINK_LIBRARIES}
        ADD_CPPLINT
        DEPENDENCIES ${DEPENDENCIES}
        LABELS
        IE
)

set_ie_threading_interface_for(${TARGET_NAME})

if(ENABLE_OV_ONNX_FRONTEND)
    target_compile_definitions(${TARGET_NAME} PRIVATE
        ENABLE_OV_ONNX_FRONTEND
        ONNX_TEST_MODELS="${TEST_MODEL_ZOO}/onnx_reader/models/")
    add_dependencies(${TARGET_NAME} openvino_onnx_frontend)
endif()

if(ENABLE_OV_PADDLE_FRONTEND)
    target_compile_definitions(${TARGET_NAME} PRIVATE
            PADDLE_TEST_MODELS="${CMAKE_CURRENT_SOURCE_DIR}/paddle_reader/models/")
    add_dependencies(${TARGET_NAME} openvino_paddle_frontend)
endif()

ie_faster_build(${TARGET_NAME}
    PCH PRIVATE "precomp.hpp"
)

# CVS-55343

file(GLOB_RECURSE legacy_tests
    "${CMAKE_CURRENT_SOURCE_DIR}/transformations/legacy/*.cpp" # CVS-55385
    "${CMAKE_CURRENT_SOURCE_DIR}/ngraph_reader/*.cpp" # CVS-55365
    "${CMAKE_CURRENT_SOURCE_DIR}/cnn_network/cnn_ngraph_impl_tests.cpp" # CVS-55375
    )

set_source_files_properties(${legacy_tests} PROPERTIES INCLUDE_DIRECTORIES
    $<TARGET_PROPERTY:inference_engine_legacy,INTERFACE_INCLUDE_DIRECTORIES>)


if(SUGGEST_OVERRIDE_SUPPORTED)
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/caching_test.cpp
                                PROPERTIES COMPILE_OPTIONS -Wno-suggest-override)
endif()

include(CMakeParseArguments)

#
# ie_headers_compilation_with_custom_flags(TEST_SUFFIX <prefix>
#                                          [FLAGS <flags>]
#                                          [PLUGIN_API]
#                                          [DEFINITIONS <definitions>]
#                                          [HEADERS_TO_SKIP <skip headers>]
#                                          [CXX_STANDARD <number>])
#
# Tests compilation with modern flags
#
function(ie_headers_compilation_with_custom_flags)
    set(options PLUGIN_API)
    set(oneValueArgs FLAGS TEST_SUFFIX CXX_STANDARD)
    set(multiValueArgs DEFINITIONS HEADERS_TO_SKIP)
    cmake_parse_arguments(IE_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(IE_TEST_PLUGIN_API)
        set(include_dirs
            "${OpenVINO_SOURCE_DIR}/src/inference/dev_api/")
    else()
        set(include_dirs
            "${OpenVINO_SOURCE_DIR}/src/core/include"
            "${OpenVINO_SOURCE_DIR}/src/inference/include/ie"
            "${OpenVINO_SOURCE_DIR}/src/inference/include/")
    endif()

    foreach(include_dir IN LISTS include_dirs)
        file(GLOB_RECURSE header_files_local RELATIVE "${include_dir}" "${include_dir}/*.hpp")
        list(APPEND header_files ${header_files_local})
    endforeach()

    if(NOT IE_TEST_CXX_STANDARD)
        set(IE_TEST_CXX_STANDARD ${CMAKE_CXX_STANDARD})
    endif()

    if(NOT IE_TEST_PLUGIN_API)
        list(APPEND IE_TEST_HEADERS_TO_SKIP "_tbl.hpp")
        if(NOT OPENCL_HEADERS_DIR)
            list(APPEND IE_TEST_HEADERS_TO_SKIP "gpu/gpu_ocl_wrapper.hpp"
                                                "gpu/gpu_context_api_ocl.hpp"
                                                "gpu/gpu_context_api_va.hpp"
                                                "gpu/gpu_context_api_dx.hpp"
                                                "gpu/gpu_ocl_wrapper.hpp"
                                                "openvino/runtime/intel_gpu/ocl/ocl_wrapper.hpp"
                                                "openvino/runtime/intel_gpu/ocl/ocl.hpp"
                                                "openvino/runtime/intel_gpu/ocl/va.hpp"
                                                "openvino/runtime/intel_gpu/ocl/dx.hpp")
        endif()
        if(NOT WIN32)
            list(APPEND IE_TEST_HEADERS_TO_SKIP "gpu/gpu_context_api_dx.hpp"
                                                "openvino/runtime/intel_gpu/ocl/dx.hpp")
        endif()
        if(NOT LIBVA_FOUND)
            list(APPEND IE_TEST_HEADERS_TO_SKIP "gpu/gpu_context_api_va.hpp"
                                                "openvino/runtime/intel_gpu/ocl/va.hpp")
        endif()
    endif()

    set(content "\n")
    foreach(header_file IN LISTS header_files)
        # skip user-passed headers
        set(skip_current_file OFF)
        foreach(skip_file IN LISTS IE_TEST_HEADERS_TO_SKIP)
            if("${header_file}" MATCHES "${skip_file}")
                set(skip_current_file ON)
            endif()
        endforeach()
        if(skip_current_file)
            continue()
        endif()

        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND
            CMAKE_CXX_COMPILER_VERSION VERSION_LESS "4.9")
            continue()
        endif()
        set(content "#include <${header_file}>\n${content}")
    endforeach()
    set(source_file "${CMAKE_CURRENT_BINARY_DIR}/modern_flags_${IE_TEST_TEST_SUFFIX}.cpp")
    file(REMOVE ${source_file})
    file(GENERATE OUTPUT ${source_file} CONTENT ${content})

    set(target_name ieFuncTestsCompilation${IE_TEST_TEST_SUFFIX})
    add_library(${target_name} OBJECT ${source_file})
    target_include_directories(${target_name} PRIVATE "${IE_TEST_INCLUDE_DIRECTORY}"
        $<TARGET_PROPERTY:inference_engine,INTERFACE_INCLUDE_DIRECTORIES>)
    target_compile_definitions(${target_name} PRIVATE $<TARGET_PROPERTY:inference_engine,INTERFACE_COMPILE_DEFINITIONS>)

    if(IE_TEST_PLUGIN_API)
        # ngraph, pugixml::static, openvino::itt, openvino_gapi_preproc headers
        target_include_directories(${target_name} SYSTEM PRIVATE
            $<TARGET_PROPERTY:pugixml::static,INTERFACE_INCLUDE_DIRECTORIES>
            $<TARGET_PROPERTY:openvino::itt,INTERFACE_INCLUDE_DIRECTORIES>)
        target_include_directories(${target_name} PRIVATE
            $<TARGET_PROPERTY:inference_engine_plugin_api,INTERFACE_INCLUDE_DIRECTORIES>)
        target_compile_definitions(${target_name} PRIVATE
            $<TARGET_PROPERTY:inference_engine_plugin_api,INTERFACE_COMPILE_DEFINITIONS>)
    else()
        # OpenCL headers if any
        if(OPENCL_HEADERS_DIR)
            target_include_directories(${target_name} SYSTEM PRIVATE ${OPENCL_HEADERS_DIR})
        endif()
    endif()

    # To include TBB headers as system
    set_ie_threading_interface_for(${target_name})

    # To avoid further TBB find_package action in next call of this function. Some version of TBB
    # has an issue with cmake config which lead to fail in case of multiple call of find_package
    # from one cmake script file.
    set(TBB_FOUND ${TBB_FOUND} PARENT_SCOPE)
    set(TBB_IMPORTED_TARGETS ${TBB_IMPORTED_TARGETS} PARENT_SCOPE)
    set(TBB_VERSION ${TBB_VERSION} PARENT_SCOPE)

    set_target_properties(${target_name} PROPERTIES
                          CXX_STANDARD ${IE_TEST_CXX_STANDARD}
                          CXX_STANDARD_REQUIRED OFF)

    if(IE_TEST_FLAGS)
        set_target_properties(${target_name} PROPERTIES
                              COMPILE_OPTIONS "${IE_TEST_FLAGS}")
    endif()

    if(IE_TEST_DEFINITIONS)
        target_compile_definitions(${target_name} PRIVATE ${IE_TEST_DEFINITIONS})
    endif()

    if(WIN32)
        set_target_properties(${target_name} PROPERTIES COMPILE_PDB_NAME ${target_name})
    endif()

    add_dependencies(${TARGET_NAME} ${target_name})
endfunction()

#
# Public headers tests
#

ie_headers_compilation_with_custom_flags(TEST_SUFFIX Cxx17
                                         CXX_STANDARD 17)

if(CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 8.0)
    ie_headers_compilation_with_custom_flags(TEST_SUFFIX Cxx20
                                             CXX_STANDARD 20)
endif()

if(UNIX)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
        ie_headers_compilation_with_custom_flags(TEST_SUFFIX WarningsAreErrors
                                                 FLAGS -Wremarks -Wcheck -Werror-all -Werror -Wall)
    elseif(BUILD_SHARED_LIBS)
        # fix warnings wit RTTI
        # ie_headers_compilation_with_custom_flags(TEST_SUFFIX Pedantic
        #                                          FLAGS -Wpedantic)
    endif()
else()
    ie_headers_compilation_with_custom_flags(TEST_SUFFIX WindowsAreErrors
                                             HEADERS_TO_SKIP "gpu/gpu_ocl_wrapper.hpp"
                                                             "gpu/gpu_context_api_ocl.hpp"
                                                             "gpu/gpu_context_api_dx.hpp"
                                             FLAGS /we4996 /W4 /WX)
endif()

# compilation with -Wweak-vtables
# if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
#     ie_headers_compilation_with_custom_flags(TEST_SUFFIX WeakTables FLAGS -Wweak-vtables)
# endif()

#
# Plugin API headers tests
#

ie_headers_compilation_with_custom_flags(TEST_SUFFIX PluginApiCxx17
                                         CXX_STANDARD 17 PLUGIN_API)

if(CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 8.0)
    if(TBB_VERSION VERSION_EQUAL 2021.2)
        # TBB 2021.2 has a bug in headers with CXX_STANDARD 20
        # see https://github.com/oneapi-src/oneTBB/issues/379
        set(skip_headers_list HEADERS_TO_SKIP "threading/ie_thread_local.hpp")
    endif()
    ie_headers_compilation_with_custom_flags(TEST_SUFFIX PluginApiCxx20
                                             ${skip_headers_list}
                                             CXX_STANDARD 20 PLUGIN_API)
endif()

if(UNIX)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
        ie_headers_compilation_with_custom_flags(TEST_SUFFIX PluginApiWarningsAreErrors
                                                 FLAGS -Werror-all -Werror -Wall
                                                 PLUGIN_API)
    elseif(BUILD_SHARED_LIBS)
        # fix warnings wit RTTI
        # ie_headers_compilation_with_custom_flags(TEST_SUFFIX PluginApiPedantic
        #                                          FLAGS -Wpedantic
        #                                          PLUGIN_API)
    endif()
else()
    ie_headers_compilation_with_custom_flags(TEST_SUFFIX PluginApiWindowsAreErrors
                                             FLAGS /we4996 /W4 /WX
                                             PLUGIN_API)
endif()

# ir serialization functional tests variables
target_compile_definitions(${TARGET_NAME} PRIVATE
    IR_SERIALIZATION_MODELS_PATH="${TEST_MODEL_ZOO}/ir_serialization/models/")
