# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

#------------------------------------------------------------------------------
# Configure and install libonnx ...
#------------------------------------------------------------------------------

set(OV_ONNX_NAMESPACE openvino_onnx)

# Apply local patches to the pinned ONNX submodule.
#
# not_equal_to_ctad.patch: ONNX 1.22 uses 'std::not_equal_to()' (relying on
# CTAD with no deduction guide), which GCC 8 (e.g. RHEL 8.10's system compiler)
# rejects with "cannot deduce template arguments for 'not_equal_to' from ()".
# The patch rewrites it to the portable 'std::not_equal_to<>()'. Newer compilers
# accept the upstream source as-is, so the patch is only needed for GCC < 9.
#
# TODO: remove this patch (and the patches directory) once GCC 8 support is
# dropped, or once the fix is upstreamed into the pinned ONNX version. Patches
# should be avoided in general and kept as narrow and short-lived as possible.
#
# Patches are listed explicitly (not globbed) so it is clear which patch applies
# under which condition. They are applied idempotently so re-configuration and
# incremental builds are safe.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
    set(_onnx_patches "${CMAKE_CURRENT_SOURCE_DIR}/patches/not_equal_to_ctad.patch")

    find_package(Git QUIET)
    if(GIT_FOUND)
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

# WA for ONNX: protobuf must be findable via an export set when CMake validates
# install(EXPORT ONNXTargets ...) at configure time.
#
# Static build (NOT BUILD_SHARED_LIBS):
# libprotobuf-lite is also added to OpenVINOTargets later by ov_install_static_deps
# (called from the ONNX frontend CMakeLists.txt via ov_add_frontend). Having it
# in BOTH ONNXTargets and OpenVINOTargets is allowed; the "exported multiple times"
# error applies only to the DEPS of a target (absl etc.), not the target itself.
#
# Shared build (BUILD_SHARED_LIBS=ON):
# ov_install_static_lib's if(NOT BUILD_SHARED_LIBS) guard AND the absence of a
# protobuf-targets set (protobuf_INSTALL=OFF) mean libprotobuf is not in any
# export set. Use ov_register_static_deps_in_export to add libprotobuf AND all
# its transitive deps to ONNXTargets (bypassing the BUILD_SHARED_LIBS guard).
if(NOT ENABLE_SYSTEM_PROTOBUF)
    if(ONNX_USE_LITE_PROTO)
        set(_onnx_protobuf_root libprotobuf-lite)
    else()
        set(_onnx_protobuf_root libprotobuf)
    endif()

    if(BUILD_SHARED_LIBS)
        # Shared build: register libprotobuf and ALL transitive deps in ONNXTargets.
        set(_onnx_pb_roots "${_onnx_protobuf_root}")
        ov_register_static_deps_in_export(_onnx_pb_roots ONNXTargets)
        unset(_onnx_pb_roots)
    else()
        # Static build: add just libprotobuf to ONNXTargets (original WA).
        # Its transitive deps (absl, utf8_validity, …) will be in OpenVINOTargets
        # via ov_install_static_deps and will be referenced as openvino::* there.
        install(TARGETS "${_onnx_protobuf_root}" EXPORT ONNXTargets
                ARCHIVE DESTINATION ${OV_CPACK_ARCHIVEDIR}
                COMPONENT ${OV_CPACK_COMP_CORE}
                ${OV_CPACK_COMP_CORE_EXCLUDE_ALL})
    endif()

    unset(_onnx_protobuf_root)
endif()
