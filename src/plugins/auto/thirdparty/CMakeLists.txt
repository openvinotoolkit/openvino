# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# Intel Platform Framework (IPF) telemetry client, built from the git submodule.
# The upstream IPF-EF ClientApi supports the MSVC toolchain only, so it is built
# solely on that toolchain. On every other toolchain the AUTO plugin falls back to
# a no-op device monitor, so IPF never blocks the AUTO plugin or OpenVINO build.

set(OV_AUTO_IPF_AVAILABLE OFF PARENT_SCOPE)

set(_ipf_ef_dir "${CMAKE_CURRENT_SOURCE_DIR}/ipf/IPF_EF")

if(NOT MSVC)
    message(STATUS "[AUTO] IPF: non-MSVC toolchain, device utilization monitoring disabled, IPF_AVAILABLE = OFF")
    return()
endif()

if(NOT EXISTS "${_ipf_ef_dir}/CMakeLists.txt")
    message(STATUS "[AUTO] IPF: submodule not found at ${_ipf_ef_dir}, device utilization monitoring disabled, IPF_AVAILABLE = OFF. "
                   "Run 'git submodule update --init --recursive' to enable it.")
    return()
endif()

add_subdirectory("${_ipf_ef_dir}" "${CMAKE_CURRENT_BINARY_DIR}/ipf" EXCLUDE_FROM_ALL)

message(STATUS "[AUTO] IPF: successfully configured, IPF_AVAILABLE = ON")
set(OV_AUTO_IPF_AVAILABLE ON PARENT_SCOPE)
