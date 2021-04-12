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

file(GLOB_RECURSE LIBRARY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc)
file(GLOB_RECURSE LIBRARY_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h)
file(GLOB_RECURSE LIBRARY_PUBLIC_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp)

set(FRONTEND_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj

source_group("src" FILES ${LIBRARY_SRC})
source_group("include" FILES ${LIBRARY_HEADERS})
source_group("public include" FILES ${LIBRARY_PUBLIC_HEADERS})

# Create shared library
add_library(frontend_manager SHARED ${LIBRARY_SRC} ${LIBRARY_HEADERS} ${LIBRARY_PUBLIC_HEADERS} ${PROTO_SRCS} ${PROTO_HDRS})
add_library(ngraph::frontend_manager ALIAS frontend_manager)

target_include_directories(frontend_manager
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../tensorflow/include
            ${CMAKE_CURRENT_SOURCE_DIR}/../onnx_editor/include
            ${CMAKE_CURRENT_SOURCE_DIR}/../onnx_import/include
            ${CMAKE_CURRENT_SOURCE_DIR}/../onnx_common/include
            ${CMAKE_CURRENT_SOURCE_DIR}/../paddlepaddle/include)

MESSAGE("HERE4: " ${CMAKE_CURRENT_SOURCE_DIR})

if(COMMAND ie_add_vs_version_file)
    ie_add_vs_version_file(NAME frontend_manager
                           FILEDESCRIPTION "Manager of OpenVINO nGraph Front Ends")
endif()

# TODO: Add other frontends to the list of dependencies temporary; in the produce version they should be discovered
# TODO: in runtime.
if (NGRAPH_ONNX_IMPORT_ENABLE)
    target_link_libraries(frontend_manager PRIVATE onnx_importer)
    target_compile_definitions(frontend_manager PRIVATE NGRAPH_ONNX_IMPORT_ENABLE)
endif()

if (NGRAPH_ONNX_EDITOR_ENABLE)
    target_link_libraries(frontend_manager PRIVATE onnx_editor)
    target_compile_definitions(frontend_manager PRIVATE NGRAPH_ONNX_EDITOR_ENABLE)
endif()

if (NGRAPH_TF_FRONTEND_ENABLE)
    target_link_libraries(frontend_manager PRIVATE tensorflow_frontend)
    target_compile_definitions(frontend_manager PRIVATE NGRAPH_TF_FRONTEND_ENABLE)
endif()

target_link_libraries(frontend_manager PRIVATE onnx_common paddlepaddle_frontend PUBLIC ngraph)

set(FRONTEND_INSTALL_INCLUDE "${NGRAPH_INSTALL_INCLUDE}/ngraph/frontend/generic")
target_include_directories(frontend_manager SYSTEM PUBLIC $<BUILD_INTERFACE:${FRONTEND_INCLUDE_DIR}>
                                                       $<INSTALL_INTERFACE:${FRONTEND_INSTALL_INCLUDE}> ${ONNX_IMPORT_INCLUDE_DIR} ${TENSORFLOW_FRONTEND_INCLUDE_DIR})
target_include_directories(frontend_manager SYSTEM PRIVATE ${NGRAPH_INCLUDE_PATH} ${ONNX_IMPORT_INCLUDE_PATH} ${ONNX_IMPORT_INCLUDE_DIR}
        ${FRONTEND_INCLUDE_DIR} ${TENSORFLOW_FRONTEND_INCLUDE_DIR})

target_include_directories(frontend_manager PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${ONNX_IMPORT_INCLUDE_DIR}  ${TENSORFLOW_FRONTEND_INCLUDE_DIR})

# TODO: Consider to remove the following block (inherited from onnx_import just in case).
if (CMAKE_CXX_COMPILER_ID MATCHES "^(Apple)?Clang$")
    target_compile_options(frontend_manager PRIVATE -Wno-undef -Wno-reserved-id-macro -Wno-switch-enum
            -Wno-invalid-offsetof -Wno-shorten-64-to-32 -Wno-unused-macros -Wno-missing-variable-declarations
            -Wno-unused-private-field -Wno-shadow -Wno-deprecated PUBLIC -Wno-undefined-func-template)
endif()

install(TARGETS frontend_manager EXPORT ngraphTargets
        RUNTIME DESTINATION ${NGRAPH_INSTALL_LIB} COMPONENT ngraph
        ARCHIVE DESTINATION ${NGRAPH_INSTALL_LIB} COMPONENT ngraph
        LIBRARY DESTINATION ${NGRAPH_INSTALL_LIB} COMPONENT ngraph)

install(DIRECTORY ${FRONTEND_INCLUDE_DIR}/frontend_manager
    DESTINATION ${FRONTEND_INSTALL_INCLUDE}
    COMPONENT ngraph
    FILES_MATCHING
        PATTERN "*.hpp"
        PATTERN "*.h"
)


if (NGRAPH_EXPORT_TARGETS_ENABLE)
    export(TARGETS frontend_manager NAMESPACE ngraph:: APPEND FILE "${NGRAPH_TARGETS_FILE}")
endif()
