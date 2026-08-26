# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# Intel Platform Framework (IPF) telemetry client (MSVC only).
# ClientApi.dll is loaded dynamically (delay-loaded) by openvino_auto_plugin, so a
# missing/incompatible ClientApi.dll only disables IPF telemetry, not the whole plugin.
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

# The native ClientApiProxy.dll shipped by the IPF driver today is Release-only. IPF's
# own runtime compatibility check (DllProxy::Open()) fails with ENOEXEC if ClientApi.dll
# does not use the same CRT "debug-ness" as ClientApiProxy.dll, regardless of whether
# OpenVINO itself is built Debug or Release.
option(ENABLE_IPF_DEBUG_WITH_RELEASE_CRT
       "Build IPF ClientApi.dll with the Release dynamic CRT (/MD) even in a Debug OpenVINO \
build, matching the Release-only ClientApiProxy.dll currently shipped by the IPF driver. \
Turn OFF once a CRT-matching (e.g. Debug) ClientApiProxy.dll is available, so ClientApi.dll \
instead matches OpenVINO's own build configuration." ON)

# Use the same configuration-specific dynamic CRT as the rest of OpenVINO.
if(DEFINED CMAKE_MSVC_RUNTIME_LIBRARY)
    set(_ov_msvc_runtime "${CMAKE_MSVC_RUNTIME_LIBRARY}")
else()
    set(_ov_msvc_runtime "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()

if(ENABLE_IPF_DEBUG_WITH_RELEASE_CRT)
    set(_ipf_client_api_runtime "MultiThreadedDLL")
else()
    set(_ipf_client_api_runtime "${_ov_msvc_runtime}")
endif()

add_subdirectory("${_ipf_ef_dir}" "${CMAKE_CURRENT_BINARY_DIR}/ipf" EXCLUDE_FROM_ALL)

if(NOT TARGET ClientApi)
    message(STATUS "[AUTO] IPF: ClientApi target not produced by submodule, device utilization monitoring disabled")
    return()
endif()

# ClientApiC.h only applies dllexport/dllimport when ClientApi_EXPORTS/ClientApi_IMPORTS
# is defined; the submodule itself defines neither, so without this the DLL would not
# export IpfCreate/IpfDestroy/etc. at all.
target_compile_definitions(ClientApi PRIVATE ClientApi_EXPORTS)
set_target_properties(ClientApi PROPERTIES EXPORT_NAME ipf_client_api_dynamic
                                            MSVC_RUNTIME_LIBRARY "${_ipf_client_api_runtime}")
add_library(openvino::ipf_client_api_dynamic ALIAS ClientApi)

# Use OpenVINO's per-config CRT for the static client; see auto/CMakeLists.txt for selection.
if(TARGET ClientApiLib)
    set_target_properties(ClientApiLib PROPERTIES EXCLUDE_FROM_ALL FALSE
                                                  EXPORT_NAME ipf_client_api_static
                                                  MSVC_RUNTIME_LIBRARY "${_ov_msvc_runtime}")
    add_library(openvino::ipf_client_api_static ALIAS ClientApiLib)

    # Keep the header-only nlohmann/json dependency out of the exported link interface.
    get_target_property(_ipf_link_interface ClientApiLib INTERFACE_LINK_LIBRARIES)
    if(_ipf_link_interface)
        list(FILTER _ipf_link_interface EXCLUDE REGEX "^(\\$<LINK_ONLY:)?nlohmann_json(::nlohmann_json)?>?$")
        set_property(TARGET ClientApiLib PROPERTY INTERFACE_LINK_LIBRARIES "${_ipf_link_interface}")
    endif()

    ov_install_static_lib(ClientApiLib ${OV_CPACK_COMP_CORE_DEV} ${OV_CPACK_COMP_CORE_DEV_EXCLUDE_ALL})
endif()

# Install ClientApi.dll only for configurations that delay-load it.
if(ENABLE_IPF_DEBUG_WITH_RELEASE_CRT)
    install(TARGETS ClientApi
            RUNTIME DESTINATION ${OV_CPACK_PLUGINSDIR}
            COMPONENT ${OV_CPACK_COMP_CORE}
            CONFIGURATIONS Debug
            ${OV_CPACK_COMP_CORE_EXCLUDE_ALL})
endif()

message(STATUS "[AUTO] IPF: successfully configured (Release: static ClientApiLib; "
               "Debug: dynamic ClientApi.dll delay-loaded when ENABLE_IPF_DEBUG_WITH_RELEASE_CRT=ON, "
               "else static ClientApiLib; ENABLE_IPF_DEBUG_WITH_RELEASE_CRT=${ENABLE_IPF_DEBUG_WITH_RELEASE_CRT})")
