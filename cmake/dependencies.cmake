# Copyright (C) 2018 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set_temp_directory(TEMP "${IE_MAIN_SOURCE_DIR}")

include(dependency_solver)

if(CMAKE_CROSSCOMPILING)
	if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "amd64.*|x86_64.*|AMD64.*")
	    set(HOST_X86_64 ON)
	endif()

	set(protoc_version "3.7.1")
	if(CMAKE_HOST_SYSTEM_NAME MATCHES Linux)
	    RESOLVE_DEPENDENCY(SYSTEM_PROTOC_ROOT
			               ARCHIVE_LIN "protoc-${protoc_version}-linux-x86_64.tar.gz"
			               TARGET_PATH "${TEMP}/protoc-${protoc_version}-linux-x86_64")
	    debug_message(STATUS "host protoc-${protoc_version} root path = " ${SYSTEM_PROTOC_ROOT})
	else()
		message(FATAL_ERROR "Unsupported host system (${CMAKE_HOST_SYSTEM_NAME}) and arch (${CMAKE_HOST_SYSTEM_PROCESSOR}) for cross-compilation")
	endif()

    reset_deps_cache(SYSTEM_PROTOC)

    message("${SYSTEM_PROTOC_ROOT}/bin")
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
