# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

#------------------------------------------------------------------------------
# Configure Google Protobuf ...
#------------------------------------------------------------------------------

set(BUILD_SHARED_LIBS OFF)

if(CMAKE_COMPILER_IS_GNUCXX OR OV_COMPILER_IS_CLANG OR (OV_COMPILER_IS_INTEL_LLVM AND UNIX))
    ov_add_compiler_flags(-Wno-all)
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 11 AND CMAKE_COMPILER_IS_GNUCXX)
        ov_add_compiler_flags(-Wno-stringop-overflow)
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # protobuf\src\google\protobuf\descriptor.cc(822) : error C4703: potentially uninitialized local pointer variable 'to_use' used
    ov_add_compiler_flags(/wd4703)
endif()

set(protobuf_VERBOSE ON)
set(protobuf_BUILD_TESTS OFF CACHE BOOL "Build tests" FORCE)
set(protobuf_BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libs" FORCE)
set(protobuf_WITH_ZLIB OFF CACHE BOOL "Build with zlib support" FORCE)
set(ABSL_PROPAGATE_CXX_STD ON CACHE BOOL "Abseil protogate CXX standard to dependent targets" FORCE)

# some projects define HAVE_ZLIB, which affects protobuf. Let's explicitly unset it
unset(HAVE_ZLIB CACHE)

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

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/protobuf/CMakeLists.txt")
    set(protobuf_dir protobuf)
else()
    set(protobuf_dir protobuf/cmake)
endif()

add_subdirectory(${protobuf_dir} EXCLUDE_FROM_ALL)
get_directory_property(protobuf_VERSION DIRECTORY ${protobuf_dir} DEFINITION protobuf_VERSION)

set(Protobuf_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/protobuf/src)

# to hide libprotobuf warnings
target_include_directories(libprotobuf-lite SYSTEM PRIVATE "${Protobuf_INCLUDE_DIRS}")

if(CMAKE_COMPILER_IS_GNUCXX OR OV_COMPILER_IS_CLANG OR OV_COMPILER_IS_INTEL_LLVM)
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

if(protobuf_VERSION VERSION_LESS 3.9)
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
        NATIVE_SOURCE_SUBDIR "${protobuf_dir}"
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
