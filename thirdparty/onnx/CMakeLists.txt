# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE OFF)

#------------------------------------------------------------------------------
# Configure and install libonnx ...
#------------------------------------------------------------------------------

set(NGRAPH_ONNX_NAMESPACE ngraph_onnx)
set(BUILD_SHARED_LIBS OFF)

if(NOT DEFINED ONNX_USE_MSVC_STATIC_RUNTIME)
    set(ONNX_USE_MSVC_STATIC_RUNTIME OFF)
endif()

macro(onnx_set_target_properties)
    target_include_directories(onnx SYSTEM PRIVATE "${Protobuf_INCLUDE_DIRS}")
    target_include_directories(onnx_proto SYSTEM PRIVATE "${Protobuf_INCLUDE_DIRS}")

    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        target_compile_options(onnx PRIVATE /WX-)
    elseif(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "^(Apple)?Clang$")
        target_compile_options(onnx PRIVATE -Wno-all)
        target_compile_options(onnx_proto PRIVATE -Wno-all -Wno-unused-variable)
    endif()

    target_compile_definitions(onnx PUBLIC ONNX_BUILD_SHARED_LIBS)
endmacro()

set(ONNX_USE_PROTOBUF_SHARED_LIBS ${BUILD_SHARED_LIBS} CACHE BOOL "Use dynamic protobuf by ONNX library" FORCE)
set(ONNX_NAMESPACE ${NGRAPH_ONNX_NAMESPACE})
set(ONNX_USE_LITE_PROTO ${NGRAPH_USE_PROTOBUF_LITE} CACHE BOOL "Use protobuf lite for ONNX library" FORCE)
set(ONNX_ML ON CACHE BOOL "Use ONNX ML" FORCE)
if(CMAKE_CROSSCOMPILING)
    set(ONNX_CUSTOM_PROTOC_EXECUTABLE ${SYSTEM_PROTOC})
endif()

add_subdirectory(onnx EXCLUDE_FROM_ALL)
onnx_set_target_properties()
