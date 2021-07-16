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

if(CMAKE_CXX_COMPILER_ID MATCHES "^(Apple)?Clang$")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-inconsistent-missing-override")
endif()

set(protobuf_BUILD_TESTS OFF CACHE BOOL "Build tests")

# When we build dll libraries. These flags make sure onnx and protobuf build with /MD, not /MT.
# These two options can't be mixed, because they requires link two imcompatiable runtime.
set(protobuf_WITH_ZLIB OFF CACHE BOOL "Build with zlib support")

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

set(protobuf_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS} CACHE BOOL "Build shared libs" FORCE)
set(protobuf_BUILD_TESTS OFF CACHE BOOL "Build tests")
set(protobuf_WITH_ZLIB OFF CACHE BOOL "Build with zlib support")

add_subdirectory(protobuf/cmake EXCLUDE_FROM_ALL)
get_directory_property(protobuf_VERSION DIRECTORY protobuf/cmake DEFINITION protobuf_VERSION)

set(Protobuf_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/protobuf/src)

if(NGRAPH_USE_PROTOBUF_LITE)
    set(Protobuf_LIBRARIES libprotobuf-lite)
else()
    set(Protobuf_LIBRARIES libprotobuf)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "^(Apple)?Clang$")
    set(_proto_libs ${Protobuf_LIBRARIES})
    if(TARGET libprotoc)
        list(APPEND _proto_libs libprotoc)
        target_compile_options(libprotoc PRIVATE -Wno-all -Wno-unused-variable)
    endif()
    set_target_properties(${_proto_libs} PROPERTIES
        CXX_VISIBILITY_PRESET default
        C_VISIBILITY_PRESET default
        VISIBILITY_INLINES_HIDDEN OFF)
    foreach(target libprotobuf libprotobuf-lite)
        target_compile_options(${target}
            PRIVATE -Wno-all -Wno-unused-variable
            PUBLIC -Wno-undef)
    endforeach()
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
    message(FATAL_ERROR "Minimum supported version of protobuf-lite library is 3.9.0")
endif()

if(ENABLE_LTO AND protobuf_VERSION VERSION_GREATER_EQUAL "3.8")
    message(WARNING "Protobuf in version 3.8.0+ can throw runtime exceptions if LTO is enabled.")
endif()

if(CMAKE_CROSSCOMPILING AND NOT PROTOC_VERSION VERSION_EQUAL protobuf_VERSION)
    message(WARNING "system protobuf version does not match with the compiled one, please update system protobuf or submodule")
endif()

# forward variables used in the other places
set(SYSTEM_PROTOC ${SYSTEM_PROTOC} PARENT_SCOPE)
set(Protobuf_LIBRARIES ${Protobuf_LIBRARIES} PARENT_SCOPE)
set(Protobuf_INCLUDE_DIRS ${Protobuf_INCLUDE_DIRS} PARENT_SCOPE)
