# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

#------------------------------------------------------------------------------
# Configure and install libonnx ...
#------------------------------------------------------------------------------

set(NGRAPH_ONNX_NAMESPACE ngraph_onnx)

macro(onnx_set_target_properties)
    target_include_directories(onnx SYSTEM PRIVATE "${Protobuf_INCLUDE_DIRS}")
    target_include_directories(onnx_proto SYSTEM PRIVATE "${Protobuf_INCLUDE_DIRS}")

    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        target_compile_options(onnx PRIVATE /WX-)
    elseif(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "^(Apple)?Clang$")
        target_compile_options(onnx PRIVATE -Wno-all)
        target_compile_options(onnx_proto PRIVATE -Wno-all -Wno-unused-variable)

        # it fixes random problems with double registration of descriptors to protobuf database
        set_target_properties(onnx_proto PROPERTIES
            CXX_VISIBILITY_PRESET default
            C_VISIBILITY_PRESET default
            VISIBILITY_INLINES_HIDDEN OFF)
    endif()

    target_compile_definitions(onnx PUBLIC ONNX_BUILD_SHARED_LIBS)

    install(TARGETS onnx_proto
        RUNTIME DESTINATION ${NGRAPH_INSTALL_LIB} COMPONENT ngraph
        ARCHIVE DESTINATION ${NGRAPH_INSTALL_LIB} COMPONENT ngraph
        LIBRARY DESTINATION ${NGRAPH_INSTALL_LIB} COMPONENT ngraph)

    export(TARGETS onnx onnx_proto NAMESPACE ngraph:: APPEND FILE "${NGRAPH_TARGETS_FILE}")
endmacro()

set(ONNX_USE_PROTOBUF_SHARED_LIBS ${BUILD_SHARED_LIBS} CACHE BOOL "Use dynamic protobuf by ONNX library")
set(ONNX_NAMESPACE ${NGRAPH_ONNX_NAMESPACE})
set(ONNX_USE_LITE_PROTO ${NGRAPH_USE_PROTOBUF_LITE} CACHE BOOL "Use protobuf lite for ONNX library")
set(ONNX_ML ON CACHE BOOL "Use ONNX ML")
if(CMAKE_CROSSCOMPILING)
    set(ONNX_CUSTOM_PROTOC_EXECUTABLE ${SYSTEM_PROTOC})
endif()

add_subdirectory(onnx EXCLUDE_FROM_ALL)
onnx_set_target_properties()
