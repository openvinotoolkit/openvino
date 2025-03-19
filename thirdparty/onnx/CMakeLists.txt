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

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # 4244 conversion from 'XXX' to 'YYY', possible loss of data
    ov_add_compiler_flags(/wd4244)
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
