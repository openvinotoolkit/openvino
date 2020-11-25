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

set(ONNX_OPSET_VERSION 13 CACHE INTERNAL "Supported version of ONNX operator set")

file(GLOB_RECURSE LIBRARY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE PUBLIC_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp)

# Remove disabled ops
list(REMOVE_ITEM LIBRARY_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/conv_integer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/matmul_integer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/qlinear_matmul.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/quant_conv.cpp
    )
list(REMOVE_ITEM PUBLIC_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/include/onnx_import/op/conv_integer.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/onnx_import/op/matmul_integer.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/onnx_import/op/qlinear_matmul.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/onnx_import/op/quant_conv.hpp
    )

set(ONNX_IMPORT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include CACHE INTERNAL "")

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj

source_group("src" FILES ${LIBRARY_SRC})
source_group("include" FILES ${PUBLIC_HEADERS})

# Create shared library
add_library(onnx_importer SHARED ${LIBRARY_SRC} ${PUBLIC_HEADERS})
add_library(ngraph::onnx_importer ALIAS onnx_importer)

if(COMMAND ie_add_vs_version_file)
    ie_add_vs_version_file(NAME onnx_importer
                           FILEDESCRIPTION "nGraph ONNX importer library")
endif()

if(COMMAND ie_faster_build)
    ie_faster_build(onnx_importer
        UNITY
        PCH PRIVATE "src/precomp.hpp"
    )
endif()

target_link_libraries(onnx_importer PRIVATE onnx onnx_proto ${Protobuf_LIBRARIES} ngraph::builder)
target_link_libraries(onnx_importer PUBLIC ngraph)

set_target_properties(onnx_importer PROPERTIES
                      CXX_VISIBILITY_PRESET hidden
                      C_VISIBILITY_PRESET hidden
                      VISIBILITY_INLINES_HIDDEN ON
                      POSITION_INDEPENDENT_CODE ON)
set(ONNX_INSTALL_INCLUDE "${NGRAPH_INSTALL_INCLUDE}/ngraph/frontend")
target_include_directories(onnx_importer SYSTEM PUBLIC $<BUILD_INTERFACE:${ONNX_IMPORT_INCLUDE_DIR}>
                                                       $<INSTALL_INTERFACE:${ONNX_INSTALL_INCLUDE}>)
target_include_directories(onnx_importer SYSTEM PRIVATE ${NGRAPH_INCLUDE_PATH} ${NGRAPH_INCLUDE_PATH}/ngraph
        ${ONNX_INCLUDE_DIR} ${ONNX_PROTO_INCLUDE_DIR} ${Protobuf_INCLUDE_DIRS})
target_include_directories(onnx_importer PRIVATE ${ONNX_IMPORT_INCLUDE_DIR}/onnx_import/core
                                                 ${ONNX_IMPORT_INCLUDE_DIR}/onnx_import/op
                                                 ${ONNX_IMPORT_INCLUDE_DIR}/onnx_import/utils)

target_compile_definitions(onnx_importer PRIVATE ONNX_OPSET_VERSION=${ONNX_OPSET_VERSION})

if(NGRAPH_USE_PROTOBUF_LITE)
    target_compile_definitions(onnx_importer PRIVATE NGRAPH_USE_PROTOBUF_LITE)
endif()

if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "^(Apple)?Clang$")
    target_compile_options(onnx_importer PRIVATE -Wno-undef -Wno-reserved-id-macro -Wno-switch-enum
            -Wno-invalid-offsetof -Wno-shorten-64-to-32 -Wno-unused-macros -Wno-missing-variable-declarations
            -Wno-unused-private-field -Wno-shadow -Wno-deprecated PUBLIC -Wno-undefined-func-template)
endif()

target_compile_definitions(onnx_importer PRIVATE ONNX_IMPORTER_DLL_EXPORTS)

install(TARGETS onnx_importer EXPORT ngraphTargets
        RUNTIME DESTINATION ${NGRAPH_INSTALL_LIB} COMPONENT ngraph
        ARCHIVE DESTINATION ${NGRAPH_INSTALL_LIB} COMPONENT ngraph
        LIBRARY DESTINATION ${NGRAPH_INSTALL_LIB} COMPONENT ngraph)

install(DIRECTORY ${ONNX_IMPORT_INCLUDE_DIR}/onnx_import
    DESTINATION ${ONNX_INSTALL_INCLUDE}
    COMPONENT ngraph
    FILES_MATCHING
        PATTERN "*.hpp"
        PATTERN "*.h"
)


if (NGRAPH_EXPORT_TARGETS_ENABLE)
    export(TARGETS onnx_importer NAMESPACE ngraph:: APPEND FILE "${NGRAPH_TARGETS_FILE}")
endif()
