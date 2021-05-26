# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ieFuncTests)

set(INCLUDES ${IE_MAIN_SOURCE_DIR}/src/inference_engine)

set(LINK_LIBRARIES
    gmock
    funcTestUtils
    ngraphFunctions
    lptNgraphFunctions
    inference_engine_transformations
    openvino::itt
    openvino::conditional_compilation
    sharedTestClasses
    inference_engine_snippets
    offline_transformations
    inference_engine
)

set(DEPENDENCIES
    mock_engine
    inference_engine_ir_reader
    inference_engine_ir_v7_reader
    template_extension
    lptNgraphFunctions
    sharedTestClasses
)

if (NOT NGRAPH_ONNX_IMPORT_ENABLE OR NGRAPH_USE_PROTOBUF_LITE)
    set(EXCLUDED_SOURCE_PATHS "${CMAKE_CURRENT_SOURCE_DIR}/onnx_reader")
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

if(NGRAPH_ONNX_IMPORT_ENABLE)
    target_compile_definitions(${TARGET_NAME} PRIVATE
        NGRAPH_ONNX_IMPORT_ENABLE
        ONNX_TEST_MODELS="${CMAKE_CURRENT_SOURCE_DIR}/onnx_reader/models/")
    add_dependencies(${TARGET_NAME} inference_engine_onnx_reader)
endif()

ie_faster_build(${TARGET_NAME}
    PCH PRIVATE "precomp.hpp"
)

# CVS-55343

file(GLOB_RECURSE legacy_tests
    "${CMAKE_CURRENT_SOURCE_DIR}/transformations/*.cpp" # CVS-55385
    "${CMAKE_CURRENT_SOURCE_DIR}/lp_transformations/*.cpp" # CVS-55376
    "${CMAKE_CURRENT_SOURCE_DIR}/ngraph_reader/*.cpp" # CVS-55365
    "${CMAKE_CURRENT_SOURCE_DIR}/cnn_network/cnn_ngraph_impl_tests.cpp" # CVS-55375
    "${CMAKE_CURRENT_SOURCE_DIR}/local_test.cpp" # CVS-55386
    )

set_source_files_properties(${legacy_tests} PROPERTIES INCLUDE_DIRECTORIES
    $<TARGET_PROPERTY:inference_engine_legacy,INTERFACE_INCLUDE_DIRECTORIES>)

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
        set(IE_TEST_INCLUDE_DIRECTORY "${IE_MAIN_SOURCE_DIR}/src/plugin_api")
    else()
        set(IE_TEST_INCLUDE_DIRECTORY "${IE_MAIN_SOURCE_DIR}/include")
    endif()

    file(GLOB_RECURSE header_files RELATIVE "${IE_TEST_INCLUDE_DIRECTORY}"
        "${IE_TEST_INCLUDE_DIRECTORY}/*")

    if(NOT IE_TEST_CXX_STANDARD)
        set(IE_TEST_CXX_STANDARD ${CMAKE_CXX_STANDARD})
    endif()

    if(NOT IE_TEST_PLUGIN_API)
        if(NOT CLDNN__OCL_ICD_INCDIRS)
            list(APPEND IE_TEST_HEADERS_TO_SKIP "gpu/gpu_ocl_wrapper.hpp"
                                                "gpu/gpu_context_api_ocl.hpp"
                                                "gpu/gpu_context_api_va.hpp"
                                                "gpu/gpu_context_api_dx.hpp")
        endif()
        if(NOT WIN32)
            list(APPEND IE_TEST_HEADERS_TO_SKIP "gpu/gpu_context_api_dx.hpp")
        endif()
        if(NOT LIBVA_FOUND)
            list(APPEND IE_TEST_HEADERS_TO_SKIP "gpu/gpu_context_api_va.hpp")
        endif()
    endif()

    set(content "\n")
    foreach(header_file IN LISTS header_files)
        # skip user-passed headers
        set(skip_current_file OFF)
        foreach(skip_file IN LISTS IE_TEST_HEADERS_TO_SKIP)
            if(header_file STREQUAL skip_file)
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
        # ngraph, pugixml, openvino::itt, inference_engine_preproc headers
        target_include_directories(${target_name} SYSTEM PRIVATE
            $<TARGET_PROPERTY:pugixml,INTERFACE_INCLUDE_DIRECTORIES>
            $<TARGET_PROPERTY:openvino::itt,INTERFACE_INCLUDE_DIRECTORIES>)
        target_include_directories(${target_name} PRIVATE
            $<TARGET_PROPERTY:${NGRAPH_LIBRARIES},INTERFACE_INCLUDE_DIRECTORIES>
            $<TARGET_PROPERTY:inference_engine_preproc,INTERFACE_INCLUDE_DIRECTORIES>)
    else()
        # OpenCL headers if any
        if(CLDNN__OCL_ICD_INCDIRS)
            target_include_directories(${target_name} SYSTEM PRIVATE ${CLDNN__OCL_ICD_INCDIRS})
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
                              COMPILE_FLAGS ${IE_TEST_FLAGS})
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

if(UNIX)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
        ie_headers_compilation_with_custom_flags(TEST_SUFFIX WarningsAreErrors
                                                 FLAGS "-Werror-all -Werror -Wall")
    else()
        ie_headers_compilation_with_custom_flags(TEST_SUFFIX Pedantic FLAGS "-Wpedantic")
    endif()
else()
    ie_headers_compilation_with_custom_flags(TEST_SUFFIX WindowsAreErrors
                                             HEADERS_TO_SKIP "gpu/gpu_ocl_wrapper.hpp"
                                                             "gpu/gpu_context_api_ocl.hpp"
                                                             "gpu/gpu_context_api_dx.hpp"
                                             FLAGS  "/we4996 /W4 /WX")
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

if(UNIX)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
        ie_headers_compilation_with_custom_flags(TEST_SUFFIX PluginApiWarningsAreErrors
                                                 FLAGS "-Werror-all -Werror -Wall"
                                                 PLUGIN_API)
    else()
        ie_headers_compilation_with_custom_flags(TEST_SUFFIX PluginApiPedantic FLAGS "-Wpedantic"
                                                 PLUGIN_API)
    endif()
else()
    ie_headers_compilation_with_custom_flags(TEST_SUFFIX PluginApiWindowsAreErrors
                                             FLAGS  "/we4996 /W4 /WX"
                                             PLUGIN_API)
endif()

# ir serialization functional tests variables
target_compile_definitions(${TARGET_NAME} PRIVATE IR_SERIALIZATION_MODELS_PATH="${CMAKE_CURRENT_SOURCE_DIR}/ir_serialization/models/")
