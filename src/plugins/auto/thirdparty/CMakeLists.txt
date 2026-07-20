# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# Intel Platform Framework (IPF) telemetry client, built from the git submodule.
# The upstream IPF-EF ClientApi supports the MSVC toolchain only, so it is built
# solely on that toolchain. On every other toolchain the AUTO plugin falls back to
# a no-op device monitor, so IPF never blocks the AUTO plugin or OpenVINO build.
#
# Only the ClientApi shared library (DLL) is used. AUTO links against its import
# library and calls into it exclusively through the plain-C ABI (ClientApiC.h), so
# IPF's own MSVC runtime library choice never has to match AUTO's. IPF requires
# ClientApi to always be built with /MD (even for Debug); this is enforced below
# regardless of what CMAKE_MSVC_RUNTIME_LIBRARY the parent build configured.

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

if(NOT TARGET ClientApi)
    message(STATUS "[AUTO] IPF: ClientApi target not produced by submodule, device utilization monitoring disabled, IPF_AVAILABLE = OFF")
    return()
endif()

set(OV_AUTO_IPF_INCLUDE_DIRS
    "${_ipf_ef_dir}/extras/c_wrapper"
    "${_ipf_ef_dir}/src/clientapi"
    "${_ipf_ef_dir}/src/shared"
    PARENT_SCOPE)
set_target_properties(ClientApi PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "")

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

message(STATUS "[AUTO] IPF: successfully configured, IPF_AVAILABLE = ON")
set(OV_AUTO_IPF_AVAILABLE ON PARENT_SCOPE)
