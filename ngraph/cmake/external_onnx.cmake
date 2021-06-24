# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include(FetchContent)

#------------------------------------------------------------------------------
# ONNX.proto definition version
#------------------------------------------------------------------------------

set(ONNX_VERSION 1.8.1)

#------------------------------------------------------------------------------
# Download and install libonnx ...
#------------------------------------------------------------------------------

set(ONNX_GIT_REPO_URL https://github.com/onnx/onnx.git)
set(ONNX_GIT_BRANCH rel-${ONNX_VERSION})
set(NGRAPH_ONNX_NAMESPACE ngraph_onnx)
set(ONNX_PATCH_FILE "${CMAKE_CURRENT_SOURCE_DIR}/cmake/patches/onnx_patch.diff")

FetchContent_Declare(
    ext_onnx
    GIT_REPOSITORY ${ONNX_GIT_REPO_URL}
    GIT_TAG ${ONNX_GIT_BRANCH}
    GIT_SHALLOW TRUE
    # apply patch to fix problems with symbols visibility for MSVC
    PATCH_COMMAND git reset --hard HEAD && git apply --ignore-space-change --ignore-whitespace --verbose ${ONNX_PATCH_FILE}
)

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

FetchContent_GetProperties(ext_onnx)
if(NOT ext_onnx_POPULATED)
    FetchContent_Populate(ext_onnx)
    set(ONNX_USE_PROTOBUF_SHARED_LIBS ${BUILD_SHARED_LIBS} CACHE BOOL "Use dynamic protobuf by ONNX library")
    set(ONNX_NAMESPACE ${NGRAPH_ONNX_NAMESPACE})
    set(ONNX_USE_LITE_PROTO ${NGRAPH_USE_PROTOBUF_LITE} CACHE BOOL "Use protobuf lite for ONNX library")
    set(ONNX_ML ON CACHE BOOL "Use ONNX ML")
    if(CMAKE_CROSSCOMPILING)
        set(ONNX_CUSTOM_PROTOC_EXECUTABLE ${SYSTEM_PROTOC})
    endif()

    add_subdirectory(${ext_onnx_SOURCE_DIR} ${ext_onnx_BINARY_DIR} EXCLUDE_FROM_ALL)
    onnx_set_target_properties()
else()
    onnx_set_target_properties()
endif()
