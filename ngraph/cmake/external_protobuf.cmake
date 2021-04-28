# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include(FetchContent)

#------------------------------------------------------------------------------
# Download and install Google Protobuf ...
#------------------------------------------------------------------------------

set(PUSH_CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE "${CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE}")
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE OFF)

if (MSVC)
    set(protobuf_MSVC_STATIC_RUNTIME OFF CACHE BOOL "")
endif()

# This version of PROTOBUF is required by Microsoft ONNX Runtime.
set(NGRAPH_PROTOBUF_GIT_REPO_URL "https://github.com/protocolbuffers/protobuf")

if(CMAKE_CROSSCOMPILING)
    find_program(SYSTEM_PROTOC protoc PATHS ENV PATH)

    if(SYSTEM_PROTOC)
        execute_process(
            COMMAND ${SYSTEM_PROTOC} --version
            OUTPUT_VARIABLE PROTOC_VERSION
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )

        string(REPLACE " " ";" PROTOC_VERSION ${PROTOC_VERSION})
        list(GET PROTOC_VERSION -1 PROTOC_VERSION)

        message("Detected system protoc version: ${PROTOC_VERSION}")

        if(${PROTOC_VERSION} VERSION_EQUAL "3.0.0")
            message(WARNING "Protobuf 3.0.0 detected switching to 3.0.2 due to bug in gmock url")
            set(PROTOC_VERSION "3.0.2")
        endif()
    else()
        message(FATAL_ERROR "System Protobuf is needed while cross-compiling")
    endif()

    set(protobuf_BUILD_PROTOC_BINARIES OFF CACHE BOOL "Build libprotoc and protoc compiler" FORCE)
elseif(NGRAPH_USE_PROTOBUF_LITE)
    set(PROTOC_VERSION "3.9.2")
    if(ENABLE_LTO)
        message(WARNING "Protobuf in version 3.8.0+ can throw runtime exceptions if LTO is enabled.")
    endif()
else()
    set(PROTOC_VERSION "3.7.1")
endif()

set(NGRAPH_PROTOBUF_GIT_TAG "v${PROTOC_VERSION}")


if (CMAKE_GENERATOR STREQUAL "Ninja")
    set(MAKE_UTIL make)
else()
    set(MAKE_UTIL $(MAKE))
endif()

if(PROTOC_VERSION VERSION_LESS "3.9" AND NGRAPH_USE_PROTOBUF_LITE)
    message(FATAL_ERROR "Minimum supported version of protobuf-lite library is 3.9.0")
else()
    if(PROTOC_VERSION VERSION_GREATER_EQUAL "3.0")
        FetchContent_Declare(
            ext_protobuf
            GIT_REPOSITORY ${NGRAPH_PROTOBUF_GIT_REPO_URL}
            GIT_TAG ${NGRAPH_PROTOBUF_GIT_TAG}
        )

        FetchContent_GetProperties(ext_protobuf)
        if(NOT ext_protobuf_POPULATED)
            FetchContent_Populate(ext_protobuf)
            set(protobuf_BUILD_TESTS OFF CACHE BOOL "Build tests")
            set(protobuf_WITH_ZLIB OFF CACHE BOOL "Build with zlib support")
            add_subdirectory(${ext_protobuf_SOURCE_DIR}/cmake ${ext_protobuf_BINARY_DIR} EXCLUDE_FROM_ALL)
        endif()
    else()
        message(FATAL_ERROR "Minimum supported version of protobuf library is 3.0.0")
    endif()

    set(Protobuf_INCLUDE_DIRS ${ext_protobuf_SOURCE_DIR}/src)
    if(NGRAPH_USE_PROTOBUF_LITE)
        set(Protobuf_LIBRARIES libprotobuf-lite)
    else()
        set(Protobuf_LIBRARIES libprotobuf)
    endif()

    if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "^(Apple)?Clang$")
        set(_proto_libs ${Protobuf_LIBRARIES})
        if(TARGET libprotoc)
            list(APPEND _proto_libs libprotoc)
            set_target_properties(libprotoc PROPERTIES
                COMPILE_FLAGS "-Wno-all -Wno-unused-variable")
        endif()
        set_target_properties(${_proto_libs} PROPERTIES
            CXX_VISIBILITY_PRESET default
            C_VISIBILITY_PRESET default
            VISIBILITY_INLINES_HIDDEN OFF)
        set_target_properties(libprotobuf libprotobuf-lite PROPERTIES
            COMPILE_FLAGS "-Wno-all -Wno-unused-variable -Wno-inconsistent-missing-override")
    endif()

    if(NGRAPH_USE_PROTOBUF_LITE)
        # if only libprotobuf-lite is used, both libprotobuf and libprotobuf-lite are built
        # libprotoc target needs symbols from libprotobuf, even in libprotobuf-lite configuration
        set_target_properties(libprotobuf PROPERTIES
            CXX_VISIBILITY_PRESET default
            C_VISIBILITY_PRESET default
            VISIBILITY_INLINES_HIDDEN OFF)
    endif()
endif()

# Now make sure we restore the original flags
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE "${PUSH_CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE}")

install(TARGETS ${Protobuf_LIBRARIES}
    RUNTIME DESTINATION ${NGRAPH_INSTALL_LIB} COMPONENT ngraph
    ARCHIVE DESTINATION ${NGRAPH_INSTALL_LIB} COMPONENT ngraph
    LIBRARY DESTINATION ${NGRAPH_INSTALL_LIB} COMPONENT ngraph)
if (NGRAPH_EXPORT_TARGETS_ENABLE)
    export(TARGETS ${Protobuf_LIBRARIES} NAMESPACE ngraph:: APPEND FILE "${NGRAPH_TARGETS_FILE}")
endif()

message(${ext_protobuf_BINARY_DIR})
#include("${ext_protobuf_BINARY_DIR}/lib/cmake/protobuf/protobuf-module.cmake")

#TODO: ---Find out the way to reuse these function from Protobuf modules ---

function(protobuf_generate)
    include(CMakeParseArguments)

    set(_options APPEND_PATH)
    set(_singleargs LANGUAGE OUT_VAR EXPORT_MACRO PROTOC_OUT_DIR)
    if(COMMAND target_sources)
        list(APPEND _singleargs TARGET)
    endif()
    set(_multiargs PROTOS IMPORT_DIRS GENERATE_EXTENSIONS)

    cmake_parse_arguments(protobuf_generate "${_options}" "${_singleargs}" "${_multiargs}" "${ARGN}")

    if(NOT protobuf_generate_PROTOS AND NOT protobuf_generate_TARGET)
        message(SEND_ERROR "Error: protobuf_generate called without any targets or source files")
        return()
    endif()

    if(NOT protobuf_generate_OUT_VAR AND NOT protobuf_generate_TARGET)
        message(SEND_ERROR "Error: protobuf_generate called without a target or output variable")
        return()
    endif()

    if(NOT protobuf_generate_LANGUAGE)
        set(protobuf_generate_LANGUAGE cpp)
    endif()
    string(TOLOWER ${protobuf_generate_LANGUAGE} protobuf_generate_LANGUAGE)

    if(NOT protobuf_generate_PROTOC_OUT_DIR)
        set(protobuf_generate_PROTOC_OUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
    endif()

    if(protobuf_generate_EXPORT_MACRO AND protobuf_generate_LANGUAGE STREQUAL cpp)
        set(_dll_export_decl "dllexport_decl=${protobuf_generate_EXPORT_MACRO}:")
    endif()

    if(NOT protobuf_generate_GENERATE_EXTENSIONS)
        if(protobuf_generate_LANGUAGE STREQUAL cpp)
            set(protobuf_generate_GENERATE_EXTENSIONS .pb.h .pb.cc)
        elseif(protobuf_generate_LANGUAGE STREQUAL python)
            set(protobuf_generate_GENERATE_EXTENSIONS _pb2.py)
        else()
            message(SEND_ERROR "Error: protobuf_generate given unknown Language ${LANGUAGE}, please provide a value for GENERATE_EXTENSIONS")
            return()
        endif()
    endif()

    if(protobuf_generate_TARGET)
        get_target_property(_source_list ${protobuf_generate_TARGET} SOURCES)
        foreach(_file ${_source_list})
            if(_file MATCHES "proto$")
                list(APPEND protobuf_generate_PROTOS ${_file})
            endif()
        endforeach()
    endif()

    if(NOT protobuf_generate_PROTOS)
        message(SEND_ERROR "Error: protobuf_generate could not find any .proto files")
        return()
    endif()

    if(protobuf_generate_APPEND_PATH)
        # Create an include path for each file specified
        foreach(_file ${protobuf_generate_PROTOS})
            get_filename_component(_abs_file ${_file} ABSOLUTE)
            get_filename_component(_abs_path ${_abs_file} PATH)
            list(FIND _protobuf_include_path ${_abs_path} _contains_already)
            if(${_contains_already} EQUAL -1)
                list(APPEND _protobuf_include_path -I ${_abs_path})
            endif()
        endforeach()
    else()
        set(_protobuf_include_path -I ${CMAKE_CURRENT_SOURCE_DIR})
    endif()

    foreach(DIR ${protobuf_generate_IMPORT_DIRS})
        get_filename_component(ABS_PATH ${DIR} ABSOLUTE)
        list(FIND _protobuf_include_path ${ABS_PATH} _contains_already)
        if(${_contains_already} EQUAL -1)
            list(APPEND _protobuf_include_path -I ${ABS_PATH})
        endif()
    endforeach()

    set(_generated_srcs_all)
    foreach(_proto ${protobuf_generate_PROTOS})
        get_filename_component(_abs_file ${_proto} ABSOLUTE)
        get_filename_component(_abs_dir ${_abs_file} DIRECTORY)
        get_filename_component(_basename ${_proto} NAME_WE)
        file(RELATIVE_PATH _rel_dir ${CMAKE_CURRENT_SOURCE_DIR} ${_abs_dir})

        set(_generated_srcs)
        message(${_rel_dir})

        foreach(_ext ${protobuf_generate_GENERATE_EXTENSIONS})
            list(APPEND _generated_srcs "${protobuf_generate_PROTOC_OUT_DIR}/${_basename}${_ext}")
        endforeach()
        list(APPEND _generated_srcs_all ${_generated_srcs})

        add_custom_command(
                OUTPUT ${_generated_srcs}
                COMMAND  protobuf::protoc
                ARGS --${protobuf_generate_LANGUAGE}_out ${_dll_export_decl}${protobuf_generate_PROTOC_OUT_DIR} ${_protobuf_include_path} ${_abs_file}
                DEPENDS ${_abs_file} protobuf::protoc
                COMMENT "Running ${protobuf_generate_LANGUAGE} protocol buffer compiler on ${_proto}"
                VERBATIM )
    endforeach()

    set_source_files_properties(${_generated_srcs_all} PROPERTIES GENERATED TRUE)
    if(protobuf_generate_OUT_VAR)
        set(${protobuf_generate_OUT_VAR} ${_generated_srcs_all} PARENT_SCOPE)
    endif()
    if(protobuf_generate_TARGET)
        target_sources(${protobuf_generate_TARGET} PRIVATE ${_generated_srcs_all})
    endif()

endfunction()

function(PROTOBUF_GENERATE_CPP SRCS HDRS)
    cmake_parse_arguments(protobuf_generate_cpp "" "EXPORT_MACRO" "" ${ARGN})

    set(_proto_files "${protobuf_generate_cpp_UNPARSED_ARGUMENTS}")
    if(NOT _proto_files)
        message(SEND_ERROR "Error: PROTOBUF_GENERATE_CPP() called without any proto files")
        return()
    endif()

    if(PROTOBUF_GENERATE_CPP_APPEND_PATH)
        set(_append_arg APPEND_PATH)
    endif()

    if(DEFINED Protobuf_IMPORT_DIRS)
        set(_import_arg IMPORT_DIRS ${Protobuf_IMPORT_DIRS})
    endif()

    set(_outvar)
    protobuf_generate(${_append_arg} LANGUAGE cpp EXPORT_MACRO ${protobuf_generate_cpp_EXPORT_MACRO} OUT_VAR _outvar ${_import_arg} PROTOS ${_proto_files})

    set(${SRCS})
    set(${HDRS})
    message(${_outvar})
    foreach(_file ${_outvar})
        if(_file MATCHES "cc$")
            list(APPEND ${SRCS} ${_file})
        else()
            list(APPEND ${HDRS} ${_file})
        endif()
    endforeach()
    set(${SRCS} ${${SRCS}} PARENT_SCOPE)
    set(${HDRS} ${${HDRS}} PARENT_SCOPE)
endfunction()