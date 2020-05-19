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

add_definitions(-DONNX_BUILD_SHARED_LIBS=ON)
add_definitions(-DONNX_NAMESPACE=${NGRAPH_ONNX_NAMESPACE})

set(CMAKE_CXX_FLAGS ${CMAKE_ORIGINAL_CXX_FLAGS})

if (WIN32)
    string(REPLACE "/W3" "/W0" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

FetchContent_Declare(
    ext_onnx
    GIT_REPOSITORY ${ONNX_GIT_REPO_URL}
    GIT_TAG ${ONNX_GIT_BRANCH}
)

FetchContent_GetProperties(ext_onnx)
if(NOT ext_onnx_POPULATED)
    FetchContent_Populate(ext_onnx)
    set(ONNX_GEN_PB_TYPE_STUBS OFF)
    set(ONNX_NAMESPACE ${NGRAPH_ONNX_NAMESPACE})
    if(CMAKE_CROSSCOMPILING)
        set(ONNX_CUSTOM_PROTOC_EXECUTABLE ${SYSTEM_PROTOC})
    endif()
    add_subdirectory(${ext_onnx_SOURCE_DIR} ${ext_onnx_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

target_include_directories(onnx PRIVATE "${Protobuf_INCLUDE_DIR}")
target_include_directories(onnx_proto PRIVATE "${Protobuf_INCLUDE_DIR}")

set(ONNX_INCLUDE_DIR ${ext_onnx_SOURCE_DIR})
set(ONNX_PROTO_INCLUDE_DIR ${ext_onnx_BINARY_DIR})

# Now make sure we restore the original CMAKE_CXX_FLAGS for the caller
set(CMAKE_CXX_FLAGS ${PUSH_CMAKE_CXX_FLAGS})
