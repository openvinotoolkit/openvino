# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# Intel Platform Framework (IPF) telemetry client from submodule.
# Built only on MSVC; other toolchains keep AUTO functional without IPF telemetry.
# IPF is disabled when OpenVINO uses static MSVC runtime (/MT, /MTd).
# Only ClientApi shared library is linked via ClientApiC.h.

set(_ipf_ef_dir "${CMAKE_CURRENT_SOURCE_DIR}/ipf/IPF_EF")

if(NOT ENABLE_IPF_CLIENT_API)
    message(STATUS "[AUTO] IPF: disabled by ENABLE_IPF_CLIENT_API=OFF")
    return()
endif()

if(NOT MSVC)
    message(STATUS "[AUTO] IPF: non-MSVC toolchain, device utilization monitoring disabled")
    return()
endif()

set(_ov_auto_msvc_static_runtime "")
if(DEFINED CMAKE_MSVC_RUNTIME_LIBRARY AND CMAKE_MSVC_RUNTIME_LIBRARY MATCHES "^MultiThreaded(Debug)?$")
    set(_ov_auto_msvc_static_runtime "${CMAKE_MSVC_RUNTIME_LIBRARY}")
elseif(CMAKE_C_FLAGS MATCHES "(^|[ /])MTd?($|[ ])")
    set(_ov_auto_msvc_static_runtime "/MT")
elseif(CMAKE_CXX_FLAGS MATCHES "(^|[ /])MTd?($|[ ])")
    set(_ov_auto_msvc_static_runtime "/MT")
endif()

if(_ov_auto_msvc_static_runtime)
    message(STATUS "[AUTO] IPF: static MSVC runtime (${_ov_auto_msvc_static_runtime}), device utilization monitoring disabled")
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

add_subdirectory("${_ipf_ef_dir}" "${CMAKE_CURRENT_BINARY_DIR}/ipf" EXCLUDE_FROM_ALL)

if(NOT TARGET ClientApi)
    message(STATUS "[AUTO] IPF: ClientApi target not produced by submodule, device utilization monitoring disabled")
    return()
endif()

set_target_properties(ClientApi PROPERTIES EXPORT_NAME ipf_client_api)
if(NOT TARGET openvino::ipf_client_api)
    add_library(openvino::ipf_client_api ALIAS ClientApi)
endif()

if(NOT BUILD_SHARED_LIBS)
    install(TARGETS ClientApi EXPORT OpenVINOTargets
            RUNTIME DESTINATION ${OV_CPACK_PLUGINSDIR} COMPONENT ${OV_CPACK_COMP_CORE} ${OV_CPACK_COMP_CORE_EXCLUDE_ALL}
            ARCHIVE DESTINATION ${OV_CPACK_ARCHIVEDIR} COMPONENT ${OV_CPACK_COMP_CORE_DEV} ${OV_CPACK_COMP_CORE_DEV_EXCLUDE_ALL})
    export(TARGETS ClientApi NAMESPACE openvino::
           APPEND FILE "${CMAKE_BINARY_DIR}/OpenVINOTargets.cmake")
else()
    install(TARGETS ClientApi
            RUNTIME DESTINATION ${OV_CPACK_PLUGINSDIR} COMPONENT ${OV_CPACK_COMP_CORE} ${OV_CPACK_COMP_CORE_EXCLUDE_ALL}
            ARCHIVE DESTINATION ${OV_CPACK_ARCHIVEDIR} COMPONENT ${OV_CPACK_COMP_CORE_DEV} ${OV_CPACK_COMP_CORE_DEV_EXCLUDE_ALL})
endif()

message(STATUS "[AUTO] IPF: successfully configured")
