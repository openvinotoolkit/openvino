# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file(GLOB_RECURSE LIBRARY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc)
file(GLOB_RECURSE LIBRARY_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h)
file(GLOB_RECURSE LIBRARY_PUBLIC_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp)

if (TARGET ext_protobuf)
#    add_dependencies(ngraph::paddlepaddle_ngraph_frontend ext_protobuf)
    add_dependencies(paddlepaddle_ngraph_frontend ext_protobuf)
endif()

set(paddlepaddle_ngraph_frontend_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj

source_group("src" FILES ${LIBRARY_SRC})
source_group("include" FILES ${LIBRARY_HEADERS})
source_group("public include" FILES ${LIBRARY_PUBLIC_HEADERS})

set(PROTO_SRCS)
set(PROTO_HDRS)

# Generate protobuf file on build time for each '.proto' file in src/proto
file(GLOB proto_files ${CMAKE_CURRENT_SOURCE_DIR}/src/proto/*.proto)

foreach(INFILE ${proto_files})
    get_filename_component(FILE_DIR ${INFILE} DIRECTORY)
    get_filename_component(FILE_WE ${INFILE} NAME_WE)
    set(OUTPUT_PB_SRC ${CMAKE_CURRENT_BINARY_DIR}/${FILE_WE}.pb.cc)
    set(OUTPUT_PB_HEADER ${CMAKE_CURRENT_BINARY_DIR}/${FILE_WE}.pb.h)
    set(GENERATED_PROTO ${INFILE})
    add_custom_command(
            OUTPUT "${OUTPUT_PB_SRC}" "${OUTPUT_PB_HEADER}"
            COMMAND protobuf::protoc ARGS --cpp_out ${CMAKE_CURRENT_BINARY_DIR} -I ${FILE_DIR} ${FILE_WE}.proto
            DEPENDS protobuf::protoc ${GENERATED_PROTO}
            COMMENT "Running C++ protocol buffer compiler on ${GENERATED_PROTO}"
            VERBATIM)
    list(APPEND PROTO_SRCS "${OUTPUT_PB_SRC}")
    list(APPEND PROTO_HDRS "${OUTPUT_PB_HEADER}")
endforeach()

add_custom_target(paddlepaddle_ngraph_frontend_proto DEPENDS ${PROTO_SRCS} ${PROTO_HDRS})

set_source_files_properties(${PROTO_HDRS} ${PROTO_HDRS} PROPERTIES GENERATED TRUE)

include_directories(${Protobuf_INCLUDE_DIRS} ${paddlepaddle_ngraph_frontend_INCLUDE_DIR})

# Create shared library
add_library(paddlepaddle_ngraph_frontend SHARED  ${LIBRARY_SRC} ${LIBRARY_HEADERS} ${LIBRARY_PUBLIC_HEADERS} ${PROTO_SRCS} ${PROTO_HDRS})

add_dependencies(paddlepaddle_ngraph_frontend libprotobuf paddlepaddle_ngraph_frontend_proto)

if(NOT WIN32)
    # Addition dependency on static libprotobuf for non-Windows
    add_dependencies(paddlepaddle_ngraph_frontend libprotobuf_static)
endif()

target_include_directories(paddlepaddle_ngraph_frontend
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${FRONTEND_INCLUDE_PATH}
            ${CMAKE_CURRENT_BINARY_DIR})

if(COMMAND ie_add_vs_version_file)
    ie_add_vs_version_file(NAME paddlepaddle_ngraph_frontend
                           FILEDESCRIPTION "FrontEnd to load and convert PaddlePaddle file format")
endif()

if(WIN32)
    target_link_libraries(paddlepaddle_ngraph_frontend PRIVATE libprotobuf PUBLIC ngraph PRIVATE ngraph::builder)
else()
    target_link_libraries(paddlepaddle_ngraph_frontend PRIVATE ${PROTOBUF_STATIC_LIB_OUTPUT} PUBLIC ngraph PRIVATE ngraph::builder)
endif()

target_link_libraries(paddlepaddle_ngraph_frontend PRIVATE frontend_manager)

add_clang_format_target(paddlepaddle_ngraph_frontend_clang FOR_TARGETS paddlepaddle_ngraph_frontend
                        EXCLUDE_PATTERNS ${PROTO_SRCS} ${PROTO_HDRS})

# TODO: Consider to remove the following block (inherited from onnx_import just in case).
if (CMAKE_CXX_COMPILER_ID MATCHES "^(Apple)?Clang$")
    target_compile_options(paddlepaddle_ngraph_frontend PRIVATE -Wno-undef -Wno-reserved-id-macro -Wno-switch-enum
            -Wno-invalid-offsetof -Wno-shorten-64-to-32 -Wno-unused-macros -Wno-missing-variable-declarations
            -Wno-unused-private-field -Wno-shadow -Wno-deprecated PUBLIC -Wno-undefined-func-template)
endif()

#install(TARGETS paddlepaddle_ngraph_frontend EXPORT ngraphTargets
#        RUNTIME DESTINATION ${NGRAPH_INSTALL_LIB} COMPONENT ngraph
#        ARCHIVE DESTINATION ${NGRAPH_INSTALL_LIB} COMPONENT ngraph
#        LIBRARY DESTINATION ${NGRAPH_INSTALL_LIB} COMPONENT ngraph)
#
#if (NGRAPH_EXPORT_TARGETS_ENABLE)
#    export(TARGETS paddlepaddle_ngraph_frontend NAMESPACE ngraph:: APPEND FILE "${NGRAPH_TARGETS_FILE}")
#endif()
