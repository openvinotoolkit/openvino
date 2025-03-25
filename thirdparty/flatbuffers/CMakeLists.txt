# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(FLATBUFFERS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(FLATBUFFERS_INSTALL OFF CACHE BOOL "" FORCE)
set(FLATBUFFERS_CPP_STD ${CMAKE_CXX_STANDARD})

# note: HOST_AARCH64 AND X86_64 are not handled for Apple explicitly, because it can work via Rosetta
if(CMAKE_CROSSCOMPILING OR (APPLE AND (HOST_X86_64 AND AARCH64)) )
    set(FLATBUFFERS_BUILD_FLATC OFF CACHE BOOL "" FORCE)
else()
    set(FLATBUFFERS_BUILD_FLATC ON CACHE BOOL "" FORCE)
endif()

# build flatc when we don't use cross-compilation

add_subdirectory(flatbuffers EXCLUDE_FROM_ALL)

# build flatc using cross-compilation

if(FLATBUFFERS_BUILD_FLATC)
    if(CMAKE_COMPILER_IS_GNUCXX)
        set_target_properties(flatc PROPERTIES COMPILE_OPTIONS "-Wno-shadow")
    endif()

    set(flatbuffers_COMPILER $<TARGET_FILE:flatc> PARENT_SCOPE)
    set(flatbuffers_DEPENDENCY flatc PARENT_SCOPE)
else()
    set(HOST_FLATC_INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}/install")

    ov_native_compile_external_project(
        TARGET_NAME host_flatc
        NATIVE_INSTALL_DIR "${HOST_FLATC_INSTALL_DIR}"
        CMAKE_ARGS "-DFLATBUFFERS_BUILD_TESTS=${FLATBUFFERS_BUILD_TESTS}"
                   "-DFLATBUFFERS_BUILD_FLATLIB=OFF"
                   "-DFLATBUFFERS_BUILD_FLATC=ON"
                   "-DFLATBUFFERS_BUILD_SHAREDLIB=OFF"
                   "-DFLATBUFFERS_CPP_STD=${FLATBUFFERS_CPP_STD}"
        NATIVE_SOURCE_SUBDIR "flatbuffers"
        NATIVE_TARGETS flatc)

    set(flatbuffers_COMPILER "${HOST_FLATC_INSTALL_DIR}/bin/flatc")
    add_executable(flatbuffers::flatc IMPORTED GLOBAL)
    set_property(TARGET flatbuffers::flatc APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
    set_target_properties(flatbuffers::flatc PROPERTIES
        IMPORTED_LOCATION_RELEASE "${flatbuffers_COMPILER}")
    set_target_properties(flatbuffers::flatc PROPERTIES
        MAP_IMPORTED_CONFIG_DEBUG Release
        MAP_IMPORTED_CONFIG_MINSIZEREL Release
        MAP_IMPORTED_CONFIG_RELWITHDEBINFO Release)
    add_dependencies(flatbuffers::flatc host_flatc)

    set(flatbuffers_DEPENDENCY host_flatc PARENT_SCOPE)
    set(flatbuffers_COMPILER "${flatbuffers_COMPILER}" PARENT_SCOPE)
endif()

# set parent scope

set(flatbuffers_LIBRARY FlatBuffers PARENT_SCOPE)
