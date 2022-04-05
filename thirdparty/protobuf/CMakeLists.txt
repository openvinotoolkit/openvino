# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

#------------------------------------------------------------------------------
# Configure Google Protobuf ...
#------------------------------------------------------------------------------

set(BUILD_SHARED_LIBS OFF)

if(SUGGEST_OVERRIDE_SUPPORTED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-suggest-override")
endif()

if(OV_COMPILER_IS_CLANG)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-inconsistent-missing-override")
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # protobuf\src\google\protobuf\descriptor.cc(822) : error C4703: potentially uninitialized local pointer variable 'to_use' used
    add_definitions("/wd4703")
endif()

set(protobuf_VERBOSE ON)
set(protobuf_BUILD_TESTS OFF CACHE BOOL "Build tests" FORCE)
set(protobuf_BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libs" FORCE)
set(protobuf_WITH_ZLIB OFF CACHE BOOL "Build with zlib support" FORCE)
set(protobuf_BUILD_CONFORMANCE OFF CACHE BOOL "" FORCE)

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
        list(APPEND _protoc_libs protoc libprotoc libprotobuf)
        set_target_properties(${_protoc_libs} PROPERTIES
            CXX_VISIBILITY_PRESET default
            C_VISIBILITY_PRESET default
            VISIBILITY_INLINES_HIDDEN OFF
            INTERPROCEDURAL_OPTIMIZATION_RELEASE OFF)
    endif()
    ov_disable_all_warnings(${_protoc_libs} libprotobuf-lite)
    set_target_properties(libprotobuf-lite PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})
endif()

if(protobuf_VERSION VERSION_LESS "3.9")
    message(FATAL_ERROR "Minimum supported version of protobuf-lite library is 3.9.0 (provided ${protobuf_VERSION})")
endif()

if(CMAKE_CROSSCOMPILING AND NOT PROTOC_VERSION VERSION_EQUAL protobuf_VERSION)
    message(WARNING "system protobuf version does not match with the compiled one, please update system protobuf or submodule")
endif()

# set to parent scope

if(SYSTEM_PROTOC)
    set(SYSTEM_PROTOC ${SYSTEM_PROTOC} PARENT_SCOPE)
    set(PROTOC_EXECUTABLE ${SYSTEM_PROTOC} PARENT_SCOPE)
else()
    set(PROTOC_EXECUTABLE $<TARGET_FILE:protoc> PARENT_SCOPE)
endif()

set(protobuf_VERSION ${protobuf_VERSION} PARENT_SCOPE)
set(Protobuf_LITE_LIBRARIES libprotobuf-lite PARENT_SCOPE)
set(Protobuf_LIBRARIES libprotobuf PARENT_SCOPE)
set(Protobuf_INCLUDE_DIRS ${Protobuf_INCLUDE_DIRS} PARENT_SCOPE)
