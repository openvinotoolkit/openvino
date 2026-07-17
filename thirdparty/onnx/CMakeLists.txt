# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

#------------------------------------------------------------------------------
# Configure and install libonnx ...
#------------------------------------------------------------------------------

set(OV_ONNX_NAMESPACE openvino_onnx)

# Apply local patches to the pinned ONNX submodule. ONNX 1.22 uses
# 'std::not_equal_to()' (relying on CTAD with no deduction guide), which older
# toolchains such as RHEL 8.10's GCC 8 reject with
# "cannot deduce template arguments for 'not_equal_to' from ()". The patch
# rewrites it to the portable 'std::not_equal_to<>()'. Newer compilers accept
# the upstream source as-is, so the patch is only needed for GCC < 9 and is
# gated accordingly to avoid modifying the submodule working tree elsewhere.
# Patches are applied idempotently so re-configuration and incremental builds
# are safe.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
    find_package(Git QUIET)
    if(GIT_FOUND)
        file(GLOB _onnx_patches "${CMAKE_CURRENT_SOURCE_DIR}/patches/*.patch")
        foreach(_onnx_patch IN LISTS _onnx_patches)
            # Skip patches that are already applied (e.g. on re-configure).
            execute_process(
                COMMAND "${GIT_EXECUTABLE}" apply --reverse --check "${_onnx_patch}"
                WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/onnx"
                RESULT_VARIABLE _onnx_patch_already_applied
                OUTPUT_QUIET ERROR_QUIET)
            if(_onnx_patch_already_applied EQUAL 0)
                message(STATUS "ONNX patch already applied: ${_onnx_patch}")
                continue()
            endif()
            execute_process(
                COMMAND "${GIT_EXECUTABLE}" apply --verbose "${_onnx_patch}"
                WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/onnx"
                RESULT_VARIABLE _onnx_patch_rv
                ERROR_VARIABLE _onnx_patch_err)
            if(NOT _onnx_patch_rv EQUAL 0)
                message(FATAL_ERROR "Failed to apply ONNX patch ${_onnx_patch}:\n${_onnx_patch_err}")
            endif()
            message(STATUS "Applied ONNX patch: ${_onnx_patch}")
        endforeach()
    else()
        message(WARNING "Git not found; ONNX source patches were not applied and the build may fail on GCC < 9")
    endif()
endif()

if(NOT DEFINED ONNX_USE_MSVC_STATIC_RUNTIME)
    set(ONNX_USE_MSVC_STATIC_RUNTIME OFF)
endif()

if(FORCE_FRONTENDS_USE_PROTOBUF)
    set(ONNX_USE_LITE_PROTO_DEFAULT OFF)
else()
    set(ONNX_USE_LITE_PROTO_DEFAULT ON)
endif()

set(PYTHON_EXECUTABLE "${Python3_EXECUTABLE}")
set(ONNX_USE_PROTOBUF_SHARED_LIBS OFF CACHE BOOL "Use dynamic protobuf by ONNX library" FORCE)
set(ONNX_NAMESPACE ${OV_ONNX_NAMESPACE})
set(ONNX_USE_LITE_PROTO ${ONNX_USE_LITE_PROTO_DEFAULT} CACHE BOOL "Use protobuf lite for ONNX library" FORCE)
set(ONNX_ML ON CACHE BOOL "Use ONNX ML" FORCE)
set(ONNX_CUSTOM_PROTOC_EXECUTABLE "${PROTOC_EXECUTABLE}")

# build targets

function(ov_onnx_build_static)
    set(BUILD_SHARED_LIBS OFF)
    add_subdirectory(onnx EXCLUDE_FROM_ALL)
endfunction()

ov_onnx_build_static()

foreach(_onnx_target onnx onnx_proto)
    target_include_directories(${_onnx_target} SYSTEM PRIVATE
        $<TARGET_PROPERTY:protobuf::libprotobuf,INTERFACE_INCLUDE_DIRECTORIES>)
endforeach()

if(MINGW)
    # OPTIONAL is a reserved word for mingw at least
    target_compile_definitions(onnx PRIVATE OPTIONAL=OPTIONAL_PLACEHOLDER)
endif()

# from onnx==1.13.1 it requires C++17 when compiling on Windows, and since onnx==1.16.0 on Linux
target_compile_features(onnx_proto PUBLIC cxx_std_17)

ov_disable_all_warnings(onnx onnx_proto)

# install

ov_install_static_lib(onnx ${OV_CPACK_COMP_CORE})
ov_install_static_lib(onnx_proto ${OV_CPACK_COMP_CORE})

# WA for ONNX: protobuf must be in the same export set of ONNX targets
# in case of protobuf::libprotobuf-lite / protobuf::libprotobuf are imported targets
if(NOT ENABLE_SYSTEM_PROTOBUF AND NOT BUILD_SHARED_LIBS)
    if(ONNX_USE_LITE_PROTO)
        set(protobuf_target_name libprotobuf-lite)
    else()
        set(protobuf_target_name libprotobuf)
    endif()

    install(TARGETS ${protobuf_target_name} EXPORT ONNXTargets
            ARCHIVE DESTINATION ${OV_CPACK_ARCHIVEDIR} COMPONENT ${OV_CPACK_COMP_CORE}
            ${OV_CPACK_COMP_CORE_EXCLUDE_ALL})
endif()
