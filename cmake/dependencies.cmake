# Copyright (C) 2018 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set_temp_directory(TEMP "${IE_MAIN_SOURCE_DIR}")

if(CMAKE_CROSSCOMPILING AND CMAKE_HOST_SYSTEM_NAME MATCHES Linux AND CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "amd64.*|x86_64.*|AMD64.*")
    set(protoc_version "3.7.1")

    RESOLVE_DEPENDENCY(SYSTEM_PROTOC_ROOT
        ARCHIVE_LIN "protoc-${protoc_version}-linux-x86_64.tar.gz"
        TARGET_PATH "${TEMP}/protoc-${protoc_version}-linux-x86_64"
        SHA256 "a1bedd5c05ca51e49f8f254faa3d7331e05b3a806c151fb111d582f154d0fee8"
    )
    debug_message(STATUS "host protoc-${protoc_version} root path = " ${SYSTEM_PROTOC_ROOT})

    reset_deps_cache(SYSTEM_PROTOC)

    find_program(
        SYSTEM_PROTOC
        NAMES protoc
        PATHS "${SYSTEM_PROTOC_ROOT}/bin"
        NO_DEFAULT_PATH)
    if(NOT SYSTEM_PROTOC)
        message(FATAL_ERROR "[ONNX IMPORTER] Missing host protoc binary")
    endif()

    update_deps_cache(SYSTEM_PROTOC "${SYSTEM_PROTOC}" "Path to host protoc for ONNX Importer")
endif()
