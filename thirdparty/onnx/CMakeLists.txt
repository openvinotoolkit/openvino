# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

#------------------------------------------------------------------------------
# Configure and install libonnx ...
#------------------------------------------------------------------------------

set(OV_ONNX_NAMESPACE openvino_onnx)

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
# The CMake rule is: for a target T in export set A, each dependency of T must
# be in AT LEAST ONE export set (not necessarily A).  However, in practice CMake
# generates the reference as the dep's own namespaced name only if the dep is in
# EXACTLY ONE set; if it is in multiple sets CMake cannot determine which
# namespace to use and errors.  The safe path is to put libprotobuf in ONNXTargets
# so onnx's dep is satisfied LOCALLY (onnx → ONNX::libprotobuf-lite, no ambiguity).
#
# Static build (NOT BUILD_SHARED_LIBS):
#   libprotobuf-lite is also added to OpenVINOTargets later by ov_install_static_deps
#   (called from the ONNX frontend CMakeLists.txt via ov_add_frontend).  Having it
#   in BOTH ONNXTargets and OpenVINOTargets is allowed; CMake 3.24+ lifted the
#   one-export-set restriction for the target itself.  The "exported multiple times"
#   error applies only to the DEPS of a target (absl etc.), not the target itself.
#
# Shared build (BUILD_SHARED_LIBS=ON):
#   ov_install_static_lib's if(NOT BUILD_SHARED_LIBS) guard AND the absence of a
#   protobuf-targets set (protobuf_INSTALL=OFF) mean libprotobuf is not in any
#   export set.  Use ov_register_static_deps_in_export to add libprotobuf AND all
#   its transitive deps to ONNXTargets (bypassing the BUILD_SHARED_LIBS guard).
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
