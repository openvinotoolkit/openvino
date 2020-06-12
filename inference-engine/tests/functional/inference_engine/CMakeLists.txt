# Copyright (C) 2019 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ieFuncTests)

add_subdirectory(extension_lib)

addIeTargetTest(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        INCLUDES
            # TODO: remove after removing `cnn_network_ngraph_imp.hpp`
            ${IE_MAIN_SOURCE_DIR}/src/inference_engine
        EXCLUDED_SOURCE_DIRS
            ${CMAKE_CURRENT_SOURCE_DIR}/extension_lib
        LINK_LIBRARIES
            gmock
            funcTestUtils
            ngraphFunctions
            inference_engine_transformations
            inference_engine_lp_transformations
        ADD_CPPLINT
        DEPENDENCIES
            extension_tests
            mock_engine
        LABELS
            IE
)

include(CMakeParseArguments)

#
# ie_headers_compilation_with_custom_flags(TEST_SUFFIX <prefix>
#                                          FLAGS <flags>
#                                          DEFINITIONS <definitions>
#                                          HEADERS_TO_SKIP <skip headers>
#                                          CXX_STANDARD <number>)
#
# Tests compilation with modern flags
#
function(ie_headers_compilation_with_custom_flags)
    set(options)
    set(oneValueArgs TEST_SUFFIX CXX_STANDARD)
    set(multiValueArgs FLAGS DEFINITIONS HEADERS_TO_SKIP)
    cmake_parse_arguments(IE_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    file(GLOB_RECURSE header_files RELATIVE "${IE_MAIN_SOURCE_DIR}/include"
        "${IE_MAIN_SOURCE_DIR}/include/*")

    if(NOT IE_TEST_CXX_STANDARD)
        set(IE_TEST_CXX_STANDARD ${CMAKE_CXX_STANDARD})
    endif()

    set(content "\n")
    foreach(header_file IN LISTS header_files)
        # OpenCL is not available
        if(header_file STREQUAL "ie_parallel.hpp" OR
           header_file STREQUAL "gpu/gpu_ocl_wrapper.hpp" OR
           header_file STREQUAL "gpu/gpu_context_api_ocl.hpp" OR
           header_file STREQUAL "gpu/gpu_context_api_dx.hpp" OR
           header_file STREQUAL "gpu/gpu_context_api_va.hpp")
            continue()
        endif()

        # Skip Windows header on Unix
        if(UNIX AND header_file STREQUAL "details/os/win_shared_object_loader.h")
            continue()
        endif()

        # Skip Unix heaeder on Windows
        if(WIN32 AND header_file STREQUAL "details/os/lin_shared_object_loader.h")
            continue()
        endif()

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
    target_include_directories(${target_name} PRIVATE $<TARGET_PROPERTY:inference_engine,INTERFACE_INCLUDE_DIRECTORIES>)
    target_compile_definitions(${target_name} PRIVATE $<TARGET_PROPERTY:inference_engine,INTERFACE_COMPILE_DEFINITIONS>)

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

    add_dependencies(${TARGET_NAME} ${target_name})
endfunction()

# compilation with c++17
ie_headers_compilation_with_custom_flags(TEST_SUFFIX Cxx17 CXX_STANDARD 17)

if(UNIX)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
        ie_headers_compilation_with_custom_flags(TEST_SUFFIX WarningsAreErrors
                                                 FLAGS "-Werror-all -Werror -Wall")
    else()
        ie_headers_compilation_with_custom_flags(TEST_SUFFIX Pedantic FLAGS -Wpedantic)
    endif()
else()
    ie_headers_compilation_with_custom_flags(TEST_SUFFIX WindowsAreErrors
                                             FLAGS "/we4996 /W4 /WX")
    ie_headers_compilation_with_custom_flags(TEST_SUFFIX Unicode
                                             DEFINITIONS UNICODE _UNICODE)
endif()

# compilation with -Wweak-vtables
# if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
#     ie_headers_compilation_with_custom_flags(TEST_SUFFIX WeakTables FLAGS -Wweak-vtables)
# endif()
