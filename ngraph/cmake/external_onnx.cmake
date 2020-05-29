# ******************************************************************************
# Copyright 2017-2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

include(FetchContent)

#------------------------------------------------------------------------------
# ONNX.proto definition version
#------------------------------------------------------------------------------

set(ONNX_VERSION 1.6.0)

#------------------------------------------------------------------------------
# Download and install libonnx ...
#------------------------------------------------------------------------------

# Since this file is going to be modifying CMAKE_CXX_FLAGS we need to preserve
# it so we won't overwrite the caller's CMAKE_CXX_FLAGS
set(PUSH_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})

set(ONNX_GIT_REPO_URL https://github.com/onnx/onnx.git)
set(ONNX_GIT_BRANCH rel-${ONNX_VERSION})
set(NGRAPH_ONNX_NAMESPACE ngraph_onnx)

set(CMAKE_CXX_FLAGS ${CMAKE_ORIGINAL_CXX_FLAGS})

FetchContent_Declare(
    ext_onnx
    GIT_REPOSITORY ${ONNX_GIT_REPO_URL}
    GIT_TAG ${ONNX_GIT_BRANCH}
)

FetchContent_GetProperties(ext_onnx)
if(NOT ext_onnx_POPULATED)
    FetchContent_Populate(ext_onnx)
    set(ONNX_NAMESPACE ${NGRAPH_ONNX_NAMESPACE})
    set(ONNX_USE_LITE_PROTO OFF CACHE BOOL "Use protobuf lite for ONNX library")
    set(ONNX_ML ON CACHE BOOL "Use ONNX ML")
    set(ONNX_BUILD_SHARED_LIBS ON CACHE BOOL "Build ONNX as a shared library")
    if(CMAKE_CROSSCOMPILING)
        set(ONNX_CUSTOM_PROTOC_EXECUTABLE ${SYSTEM_PROTOC})
    endif()
    add_subdirectory(${ext_onnx_SOURCE_DIR} ${ext_onnx_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

target_include_directories(onnx PRIVATE "${Protobuf_INCLUDE_DIRS}")
target_include_directories(onnx_proto PRIVATE "${Protobuf_INCLUDE_DIRS}")

if(MSVC)
    target_compile_options(onnx PRIVATE /WX-)
else()
    target_compile_options(onnx PRIVATE -Wno-error)
endif()

set(ONNX_INCLUDE_DIR ${ext_onnx_SOURCE_DIR})
set(ONNX_PROTO_INCLUDE_DIR ${ext_onnx_BINARY_DIR})

# Now make sure we restore the original CMAKE_CXX_FLAGS for the caller
set(CMAKE_CXX_FLAGS ${PUSH_CMAKE_CXX_FLAGS})
