# Copyright (C) 2018-2023 Intel Corporation
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

# note: HOST_AARCH64 AND X86_64 are not handled for Apple explicitly, because it can work via Rosetta
if(CMAKE_CROSSCOMPILING OR
    (APPLE AND (HOST_X86_64 AND AARCH64)) OR
    (MSVC AND (HOST_X86_64 AND (AARCH64 OR ARM))))
    set(protobuf_BUILD_PROTOC_BINARIES OFF CACHE BOOL "Build protoc binaries" FORCE)
else()
    set(protobuf_BUILD_PROTOC_BINARIES ON CACHE BOOL "Build protoc binaries" FORCE)
endif()

# When we build dll libraries. These flags make sure onnx and protobuf build with /MD, not /MT.
# These two options can't be mixed, because they requires link two incompatible runtime.
if(NOT DEFINED protobuf_MSVC_STATIC_RUNTIME)
    set(protobuf_MSVC_STATIC_RUNTIME OFF CACHE BOOL "Link protobuf to static runtime libraries" FORCE)
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

# build protoc separatelly for build system processor

if(NOT protobuf_BUILD_PROTOC_BINARIES)
    set(HOST_PROTOC_INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}/install")

    ov_native_compile_external_project(
        TARGET_NAME host_protoc
        NATIVE_INSTALL_DIR "${HOST_PROTOC_INSTALL_DIR}"
        CMAKE_ARGS "-Dprotobuf_VERBOSE=${protobuf_VERBOSE}"
                   "-Dprotobuf_BUILD_TESTS=${protobuf_BUILD_TESTS}"
                   "-Dprotobuf_WITH_ZLIB=${protobuf_WITH_ZLIB}"
        NATIVE_SOURCE_SUBDIR "protobuf/cmake"
        NATIVE_TARGETS protoc libprotobuf-lite)

    set(PROTOC_EXECUTABLE "${HOST_PROTOC_INSTALL_DIR}/bin/protoc")
    add_executable(protobuf::protoc IMPORTED GLOBAL)
    set_property(TARGET protobuf::protoc APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
    set_target_properties(protobuf::protoc PROPERTIES
        IMPORTED_LOCATION_RELEASE "${PROTOC_EXECUTABLE}")
    set_target_properties(protobuf::protoc PROPERTIES
        MAP_IMPORTED_CONFIG_DEBUG Release
        MAP_IMPORTED_CONFIG_MINSIZEREL Release
        MAP_IMPORTED_CONFIG_RELWITHDEBINFO Release)
    add_dependencies(protobuf::protoc host_protoc)

    set(PROTOC_DEPENDENCY host_protoc PARENT_SCOPE)
    set(PROTOC_EXECUTABLE "${PROTOC_EXECUTABLE}" PARENT_SCOPE)
else()
    set(PROTOC_EXECUTABLE $<TARGET_FILE:protoc> PARENT_SCOPE)
    set(PROTOC_DEPENDENCY protoc PARENT_SCOPE)
endif()

# set to parent scope

set(protobuf_VERSION ${protobuf_VERSION} PARENT_SCOPE)
set(Protobuf_LITE_LIBRARIES libprotobuf-lite PARENT_SCOPE)
set(Protobuf_LIBRARIES libprotobuf PARENT_SCOPE)
set(Protobuf_INCLUDE_DIRS ${Protobuf_INCLUDE_DIRS} PARENT_SCOPE)
