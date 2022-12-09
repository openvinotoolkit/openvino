# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(FLATBUFFERS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(FLATBUFFERS_BUILD_FLATHASH OFF CACHE BOOL "" FORCE)
set(FLATBUFFERS_INSTALL OFF CACHE BOOL "" FORCE)

# note: HOST_AARCH64 AND X86_64 are not handled for Apple explicitly, becuase it can work via Rosetta
if(CMAKE_CROSSCOMPILING OR (APPLE AND (HOST_X86_64 AND AARCH64)) )
    set(FLATBUFFERS_BUILD_FLATC OFF CACHE BOOL "" FORCE)
else()
    set(FLATBUFFERS_BUILD_FLATC ON CACHE BOOL "" FORCE)
endif()

# build library at least
add_subdirectory(flatbuffers EXCLUDE_FROM_ALL)

# build flatc separatelly for host system processor

if(NOT FLATBUFFERS_BUILD_FLATC)
    set(HOST_FLATC_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/flatbuffers")
    set(HOST_FLATC_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/host_flatc_build")
    set(HOST_FLATC_INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}/host_flatc_install")
    set(HOST_FLATC_PREFIX_DIR "${CMAKE_CURRENT_BINARY_DIR}/host_flatc_root")

    ov_native_compile_external_project(
        TARGET_NAME host_flatc
        NATIVE_SOURCE_DIR "${HOST_FLATC_SOURCE_DIR}"
        NATIVE_BUILD_DIR "${HOST_FLATC_BUILD_DIR}"
        NATIVE_INSTALL_DIR "${HOST_FLATC_INSTALL_DIR}"
        NATIVE_PREFIX_DIR "${HOST_FLATC_PREFIX_DIR}"
        CMAKE_OPTIONS "-DFLATBUFFERS_BUILD_TESTS=${FLATBUFFERS_BUILD_TESTS}"
                      "-DFLATBUFFERS_BUILD_FLATHASH=${FLATBUFFERS_BUILD_FLATHASH}")

    add_executable(flatbuffers::flatc IMPORTED GLOBAL)
    set_target_properties(flatbuffers::flatc PROPERTIES
        IMPORTED_LOCATION_RELEASE "${HOST_FLATC_INSTALL_DIR}/bin/flatc")
    add_dependencies(flatbuffers::flatc host_flatc)

    set(flatbuffers_DEPENDENCY host_flatc PARENT_SCOPE)
    set(flatbuffers_COMPILER "${HOST_FLATC_INSTALL_DIR}/bin/flatc" PARENT_SCOPE)
else()
    set(flatbuffers_COMPILER $<TARGET_FILE:flatc> PARENT_SCOPE)
    set(flatbuffers_DEPENDENCY flatc PARENT_SCOPE)
endif()

# set parent scope

set(flatbuffers_LIBRARY flatbuffers PARENT_SCOPE)
