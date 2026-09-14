# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# Intel Platform Framework (IPF) telemetry client (MSVC only).
# ClientApiLib is linked statically into openvino_auto_plugin.
# IPF requires the dynamic CRT (/MD); static CRT toolchains disable IPF.

set(_ipf_ef_dir "${CMAKE_CURRENT_SOURCE_DIR}/ipf/IPF_EF")

if(NOT ENABLE_IPF_CLIENT_API)
    message(STATUS "[AUTO] IPF: disabled by ENABLE_IPF_CLIENT_API=OFF")
    return()
endif()

if(NOT MSVC)
    message(STATUS "[AUTO] IPF: non-MSVC toolchain, device utilization monitoring disabled")
    return()
endif()

if(NOT EXISTS "${_ipf_ef_dir}/CMakeLists.txt")
    message(STATUS "[AUTO] IPF: submodule not found at ${_ipf_ef_dir}, device utilization monitoring disabled. "
                   "Run 'git submodule update --init --recursive' to enable it.")
    return()
endif()

if(TARGET nlohmann_json::nlohmann_json OR TARGET nlohmann_json)
    message(STATUS "[AUTO] IPF: reusing existing nlohmann_json target, skipping IPF's bundled FetchContent copy")
    set_property(GLOBAL PROPERTY _FetchContent_nlohmann_json_populated TRUE)
endif()

# Use the same configuration-specific dynamic CRT for ClientApiLib and the plugin.
if(DEFINED CMAKE_MSVC_RUNTIME_LIBRARY)
    set(_ov_msvc_runtime "${CMAKE_MSVC_RUNTIME_LIBRARY}")
else()
    set(_ov_msvc_runtime "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()

add_subdirectory("${_ipf_ef_dir}" "${CMAKE_CURRENT_BINARY_DIR}/ipf" EXCLUDE_FROM_ALL)

if(NOT TARGET ClientApiLib)
    message(STATUS "[AUTO] IPF: ClientApiLib target not produced by submodule, device utilization monitoring disabled")
    return()
endif()

set_target_properties(ClientApiLib PROPERTIES EXPORT_NAME ipf_client_api
                                              MSVC_RUNTIME_LIBRARY "${_ov_msvc_runtime}")
add_library(openvino::ipf_client_api ALIAS ClientApiLib)

# The submodule also defines the stand-alone ClientApi DLL, which OpenVINO does not ship.
if(TARGET ClientApi)
    set_target_properties(ClientApi PROPERTIES EXCLUDE_FROM_ALL TRUE
                                               EXCLUDE_FROM_DEFAULT_BUILD TRUE)
endif()

# Keep the header-only nlohmann/json dependency out of the exported link interface.
get_target_property(_ipf_link_interface ClientApiLib INTERFACE_LINK_LIBRARIES)
if(_ipf_link_interface)
    list(FILTER _ipf_link_interface EXCLUDE REGEX "^(\\$<LINK_ONLY:)?nlohmann_json(::nlohmann_json)?>?$")
    set_property(TARGET ClientApiLib PROPERTY INTERFACE_LINK_LIBRARIES "${_ipf_link_interface}")
endif()

# Install ClientApiLib for static OpenVINO builds.
ov_install_static_lib(ClientApiLib ${OV_CPACK_COMP_CORE_DEV} ${OV_CPACK_COMP_CORE_DEV_EXCLUDE_ALL})

message(STATUS "[AUTO] IPF: successfully configured (ClientApiLib linked statically)")
