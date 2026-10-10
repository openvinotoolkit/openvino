# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

#------------------------------------------------------------------------------
# Configure and install libonnx ...
#------------------------------------------------------------------------------

set(OV_ONNX_NAMESPACE openvino_onnx)

# Apply local patches to the pinned ONNX submodule.
#
# These patches keep the bundled ONNX source buildable with older toolchains
# (notably GCC 8 on RHEL 8.10, whose libstdc++ predates several C++17 library
# features ONNX relies on). Newer compilers accept the upstream source as-is,
# so every patch is gated on the exact compiler version that needs it.
#
#   not_equal_to_ctad.patch     - ONNX 1.23.1 uses 'std::not_equal_to()', which
#                                 GCC < 9 rejects (it attempts class template
#                                 argument deduction instead of using the
#                                 default <void> argument). Rewritten to the
#                                 portable 'std::not_equal_to<>()'.
#   printer_to_chars_fallback.patch - printer.cc formats floats with
#                                 'std::to_chars', whose floating-point
#                                 overloads are missing on some standard
#                                 libraries: libstdc++ ships them only from
#                                 GCC 11, and libc++ marks them available only
#                                 on macOS 13.3+. Adds an snprintf round-trip
#                                 fallback (guarded so the integer path never
#                                 instantiates the missing overload).
#
# TODO: remove these patches (and the patches directory) once the affected
# toolchains are dropped, or once the fixes are upstreamed into the pinned ONNX
# version. Patches should be avoided in general and kept as narrow and
# short-lived as possible.
#
# Patches are listed explicitly (not globbed) so it is clear which patch applies
# under which condition. They are applied idempotently so re-configuration and
# incremental builds are safe.
function(ov_onnx_apply_patch patch_file)
    find_package(Git QUIET)
    if(NOT GIT_FOUND)
        message(WARNING "Git not found; ONNX source patch '${patch_file}' was not applied and the build may fail")
        return()
    endif()
    # Skip patches that are already applied (e.g. on re-configure).
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" apply --reverse --check "${patch_file}"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/onnx"
        RESULT_VARIABLE _already_applied
        OUTPUT_QUIET ERROR_QUIET)
    if(_already_applied EQUAL 0)
        message(STATUS "ONNX patch already applied: ${patch_file}")
        return()
    endif()
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" apply --verbose "${patch_file}"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/onnx"
        RESULT_VARIABLE _apply_rv
        ERROR_VARIABLE _apply_err)
    if(NOT _apply_rv EQUAL 0)
        message(FATAL_ERROR "Failed to apply ONNX patch ${patch_file}:\n${_apply_err}")
    endif()
    message(STATUS "Applied ONNX patch: ${patch_file}")
endfunction()

# std::not_equal_to() CTAD: only GCC < 9 needs it.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
    ov_onnx_apply_patch("${CMAKE_CURRENT_SOURCE_DIR}/patches/not_equal_to_ctad.patch")
endif()

# Floating-point std::to_chars is unavailable on libstdc++ < 11 (GCC < 11) and
# on libc++ when targeting macOS < 13.3 (Apple clang). Apply the fallback for
# those toolchains.
set(_onnx_need_to_chars_fallback OFF)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 11)
    set(_onnx_need_to_chars_fallback ON)
elseif(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(_onnx_need_to_chars_fallback ON)
endif()
if(_onnx_need_to_chars_fallback)
    ov_onnx_apply_patch("${CMAKE_CURRENT_SOURCE_DIR}/patches/printer_to_chars_fallback.patch")
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
set(ONNX_INSTALL OFF CACHE BOOL "Install ONNX targets, headers and CMake config files" FORCE)
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
