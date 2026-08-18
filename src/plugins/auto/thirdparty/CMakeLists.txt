# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# Intel Platform Framework (IPF) telemetry client from submodule (MSVC only).
# The static ClientApiLib is linked directly into openvino_auto_plugin so that no
# separate ClientApi.dll has to be built, signed and packaged with OpenVINO.
# IPF requires a dynamic CRT (/MD); the static CRT toolchain sets
# ENABLE_IPF_CLIENT_API=OFF.

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

# IPF pins its own directory scope to the release dynamic CRT. That is required for the
# stand-alone ClientApi.dll, but objects linked into openvino_auto_plugin must use the very
# same CRT as the plugin, otherwise MSVC rejects the link (LNK2038, _ITERATOR_DEBUG_LEVEL).
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

# nlohmann/json is header-only and is consumed by the IPF sources only. Keeping it in the link
# interface of a static library (CMake records even PRIVATE dependencies there as $<LINK_ONLY:...>)
# would drag a target that is not part of OpenVINOTargets into the export set.
get_target_property(_ipf_link_interface ClientApiLib INTERFACE_LINK_LIBRARIES)
if(_ipf_link_interface)
    list(FILTER _ipf_link_interface EXCLUDE REGEX "nlohmann_json")
    set_property(TARGET ClientApiLib PROPERTY INTERFACE_LINK_LIBRARIES "${_ipf_link_interface}")
endif()

# Only static OpenVINO builds need the archive: there the plugin itself is a static library,
# so downstream consumers have to resolve the IPF symbols at their own link step.
if(NOT BUILD_SHARED_LIBS)
    install(TARGETS ClientApiLib EXPORT OpenVINOTargets
            ARCHIVE DESTINATION ${OV_CPACK_ARCHIVEDIR} COMPONENT ${OV_CPACK_COMP_CORE_DEV} ${OV_CPACK_COMP_CORE_DEV_EXCLUDE_ALL})
    export(TARGETS ClientApiLib NAMESPACE openvino::
           APPEND FILE "${CMAKE_BINARY_DIR}/OpenVINOTargets.cmake")
endif()

message(STATUS "[AUTO] IPF: successfully configured (ClientApiLib linked statically)")
