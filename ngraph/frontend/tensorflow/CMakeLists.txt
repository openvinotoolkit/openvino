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

find_package(Protobuf REQUIRED IMPORTED)
include_directories(${Protobuf_INCLUDE_DIRS})

set(TENSORFLOW_FRONTEND_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj

source_group("src" FILES ${LIBRARY_SRC})
source_group("include" FILES ${LIBRARY_HEADERS})
source_group("public include" FILES ${LIBRARY_PUBLIC_HEADERS})

set(PROTOBUF_GENERATE_CPP_APPEND_PATH ON)
file(GLOB proto_files ${CMAKE_CURRENT_SOURCE_DIR}/src/proto/*.proto)
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${proto_files})

include_directories(${Protobuf_INCLUDE_DIRS})

# Create shared library
add_library(tensorflow_frontend SHARED  ${LIBRARY_SRC} ${LIBRARY_HEADERS} ${LIBRARY_PUBLIC_HEADERS} ${PROTO_SRCS} ${PROTO_HDRS})
add_library(ngraph::tensorflow_frontend ALIAS tensorflow_frontend)

target_include_directories(tensorflow_frontend PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../generic/include ${CMAKE_CURRENT_BINARY_DIR})

if(COMMAND ie_add_vs_version_file)
    ie_add_vs_version_file(NAME tensorflow_frontend
                           FILEDESCRIPTION "FrontEnd to load and convert Tensorflow file format")
endif()

target_link_libraries(tensorflow_frontend PRIVATE ${Protobuf_LIBRARIES} PUBLIC ngraph)

# TODO: Consider to remove the following block (inherited from onnx_import just in case).
if (CMAKE_CXX_COMPILER_ID MATCHES "^(Apple)?Clang$")
    target_compile_options(tensorflow_frontend PRIVATE -Wno-undef -Wno-reserved-id-macro -Wno-switch-enum
            -Wno-invalid-offsetof -Wno-shorten-64-to-32 -Wno-unused-macros -Wno-missing-variable-declarations
            -Wno-unused-private-field -Wno-shadow -Wno-deprecated PUBLIC -Wno-undefined-func-template)
endif()

install(TARGETS tensorflow_frontend EXPORT ngraphTargets
        RUNTIME DESTINATION ${NGRAPH_INSTALL_LIB} COMPONENT ngraph
        ARCHIVE DESTINATION ${NGRAPH_INSTALL_LIB} COMPONENT ngraph
        LIBRARY DESTINATION ${NGRAPH_INSTALL_LIB} COMPONENT ngraph)

if (NGRAPH_EXPORT_TARGETS_ENABLE)
    export(TARGETS tensorflow_frontend NAMESPACE ngraph:: APPEND FILE "${NGRAPH_TARGETS_FILE}")
endif()
