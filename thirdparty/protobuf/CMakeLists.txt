# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

#------------------------------------------------------------------------------
# Configure Google Protobuf ...
#------------------------------------------------------------------------------

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE OFF)
set(BUILD_SHARED_LIBS OFF)

if(SUGGEST_OVERRIDE_SUPPORTED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-suggest-override")
endif()

if(OV_COMPILER_IS_CLANG)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-inconsistent-missing-override")
endif()

set(protobuf_BUILD_TESTS OFF CACHE BOOL "Build tests")
set(protobuf_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS} CACHE BOOL "Build shared libs" FORCE)
set(protobuf_WITH_ZLIB OFF CACHE BOOL "Build with zlib support")

# When we build dll libraries. These flags make sure onnx and protobuf build with /MD, not /MT.
# These two options can't be mixed, because they requires link two imcompatiable runtime.
if(NOT DEFINED protobuf_MSVC_STATIC_RUNTIME)
    set(protobuf_MSVC_STATIC_RUNTIME OFF CACHE BOOL "Link protobuf to static runtime libraries" FORCE)
endif()

if(CMAKE_CROSSCOMPILING)
    find_host_program(SYSTEM_PROTOC protoc PATHS ENV PATH)

    if(SYSTEM_PROTOC)
        execute_process(
            COMMAND ${SYSTEM_PROTOC} --version
            OUTPUT_VARIABLE PROTOC_VERSION
            OUTPUT_STRIP_TRAILING_WHITESPACE)

        string(REPLACE " " ";" PROTOC_VERSION ${PROTOC_VERSION})
        list(GET PROTOC_VERSION -1 PROTOC_VERSION)

        message("Detected system protoc version: ${PROTOC_VERSION}")
    else()
        message(FATAL_ERROR "System Protobuf is needed while cross-compiling")
    endif()

    set(protobuf_BUILD_PROTOC_BINARIES OFF CACHE BOOL "Build libprotoc and protoc compiler" FORCE)
endif()

add_subdirectory(protobuf/cmake EXCLUDE_FROM_ALL)
get_directory_property(protobuf_VERSION DIRECTORY protobuf/cmake DEFINITION protobuf_VERSION)

set(Protobuf_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/protobuf/src)

# to hide libprotobuf warnings
target_include_directories(libprotobuf-lite SYSTEM PRIVATE "${Protobuf_INCLUDE_DIRS}")

if(CMAKE_COMPILER_IS_GNUCXX OR OV_COMPILER_IS_CLANG OR
   CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    if(protobuf_BUILD_PROTOC_BINARIES)
        set_target_properties(protoc libprotoc PROPERTIES
            CXX_VISIBILITY_PRESET default
            C_VISIBILITY_PRESET default
            VISIBILITY_INLINES_HIDDEN OFF)
        list(APPEND _protobuf_libs protoc libprotoc)
    endif()
    ov_disable_all_warnings(${_protobuf_libs} libprotobuf libprotobuf-lite)
endif()

if(NGRAPH_USE_PROTOBUF_LITE)
    # if only libprotobuf-lite is used, both libprotobuf and libprotobuf-lite are built
    # libprotoc target needs symbols from libprotobuf, even in libprotobuf-lite configuration
    set_target_properties(libprotobuf PROPERTIES
        CXX_VISIBILITY_PRESET default
        C_VISIBILITY_PRESET default
        VISIBILITY_INLINES_HIDDEN OFF)
endif()

if(protobuf_VERSION VERSION_LESS "3.9" AND NGRAPH_USE_PROTOBUF_LITE)
    message(FATAL_ERROR "Minimum supported version of protobuf-lite library is 3.9.0 (provided ${protobuf_VERSION})")
endif()

if(ENABLE_LTO AND protobuf_VERSION VERSION_GREATER_EQUAL "3.8")
    message(WARNING "Protobuf in version 3.8.0+ can throw runtime exceptions if LTO is enabled.")
endif()

if(CMAKE_CROSSCOMPILING AND NOT PROTOC_VERSION VERSION_EQUAL protobuf_VERSION)
    message(WARNING "system protobuf version does not match with the compiled one, please update system protobuf or submodule")
endif()

# forward variables used in the other places
if(SYSTEM_PROTOC)
    set(SYSTEM_PROTOC ${SYSTEM_PROTOC} PARENT_SCOPE)
    set(PROTOC_EXECUTABLE ${SYSTEM_PROTOC} PARENT_SCOPE)
else()
    set(PROTOC_EXECUTABLE $<TARGET_FILE:protoc> PARENT_SCOPE)
endif()

set(protobuf_VERSION ${protobuf_VERSION} PARENT_SCOPE)
set(Protobuf_LITE_LIBRARIES libprotobuf-lite PARENT_SCOPE)
set(Protobuf_LIBRARIES libprotobuf-lite PARENT_SCOPE)
set(Protobuf_INCLUDE_DIRS ${Protobuf_INCLUDE_DIRS} PARENT_SCOPE)
