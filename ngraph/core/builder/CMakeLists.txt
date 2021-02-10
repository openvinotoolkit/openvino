# ******************************************************************************
# Copyright 2017-2021 Intel Corporation
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

set(TARGET_NAME "ngraph_builders")

file(GLOB_RECURSE LIBRARY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE PUBLIC_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp)

set(BUILDER_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/)

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj

source_group("src" FILES ${LIBRARY_SRC})
source_group("include" FILES ${PUBLIC_HEADERS})

# Create shared library
add_library(${TARGET_NAME} STATIC ${LIBRARY_SRC} ${PUBLIC_HEADERS})

if(COMMAND ie_faster_build)
    ie_faster_build(${TARGET_NAME}
        UNITY
        PCH PRIVATE "src/precomp.hpp"
    )
endif()

# Defines macro in C++ to load backend plugin
target_include_directories(${TARGET_NAME} PUBLIC ${BUILDER_INCLUDE_DIR}
                                          PRIVATE ${NGRAPH_INCLUDE_PATH})

# Add an alias so that library can be used inside the build tree, e.g. when testing
add_library(ngraph::builder ALIAS ${TARGET_NAME})

# developer package

openvino_developer_export_targets(COMPONENT ngraph TARGETS ngraph::builder)
