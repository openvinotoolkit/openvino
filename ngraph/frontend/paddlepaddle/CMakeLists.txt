# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "paddlepaddle_ngraph_frontend")

file(GLOB_RECURSE LIBRARY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc)
file(GLOB_RECURSE LIBRARY_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h)
file(GLOB_RECURSE LIBRARY_PUBLIC_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp)

if (TARGET ext_protobuf)
#    add_dependencies(ngraph::paddlepaddle_ngraph_frontend ext_protobuf)
    add_dependencies(${TARGET_NAME} ext_protobuf)
endif()

set(${TARGET_NAME}_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

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

add_custom_target(${TARGET_NAME}_proto DEPENDS ${PROTO_SRCS} ${PROTO_HDRS})

set_source_files_properties(${PROTO_SRCS} ${PROTO_HDRS} PROPERTIES GENERATED TRUE)

# Disable all warnings for generated code
set_source_files_properties(${PROTO_SRCS} ${PROTO_HDRS} PROPERTIES COMPILE_FLAGS -w)

include_directories(${Protobuf_INCLUDE_DIRS} ${${TARGET_NAME}_INCLUDE_DIR})

# Create shared library
add_library(${TARGET_NAME} SHARED  ${LIBRARY_SRC} ${LIBRARY_HEADERS} ${LIBRARY_PUBLIC_HEADERS} ${PROTO_SRCS} ${PROTO_HDRS})

add_dependencies(${TARGET_NAME} libprotobuf paddlepaddle_ngraph_frontend_proto)

if(NOT WIN32)
    # Addition dependency on static libprotobuf for non-Windows
    add_dependencies(${TARGET_NAME} libprotobuf_static)
endif()

target_include_directories(${TARGET_NAME}
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${FRONTEND_INCLUDE_PATH}
            ${CMAKE_CURRENT_BINARY_DIR})

if(COMMAND ie_add_vs_version_file)
    ie_add_vs_version_file(NAME ${TARGET_NAME}
                           FILEDESCRIPTION "FrontEnd to load and convert PaddlePaddle file format")
endif()

if(WIN32)
    target_link_libraries(${TARGET_NAME} PRIVATE libprotobuf PUBLIC ngraph PRIVATE ngraph::builder)
else()
    target_link_libraries(${TARGET_NAME} PRIVATE ${PROTOBUF_STATIC_LIB_OUTPUT} PUBLIC ngraph PRIVATE ngraph::builder)
endif()

target_link_libraries(${TARGET_NAME} PRIVATE frontend_manager)

add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME}
                        EXCLUDE_PATTERNS ${PROTO_SRCS} ${PROTO_HDRS})

# TODO: Consider to remove the following block (inherited from onnx_import just in case).
if (CMAKE_CXX_COMPILER_ID MATCHES "^(Apple)?Clang$")
    target_compile_options(${TARGET_NAME} PRIVATE -Wno-undef -Wno-reserved-id-macro -Wno-switch-enum
            -Wno-invalid-offsetof -Wno-shorten-64-to-32 -Wno-unused-macros -Wno-missing-variable-declarations
            -Wno-unused-private-field -Wno-shadow -Wno-deprecated PUBLIC -Wno-undefined-func-template)
endif()

install(TARGETS ${TARGET_NAME} EXPORT ngraphTargets
        RUNTIME DESTINATION ${NGRAPH_INSTALL_LIB} COMPONENT ngraph
        ARCHIVE DESTINATION ${NGRAPH_INSTALL_LIB} COMPONENT ngraph
        LIBRARY DESTINATION ${NGRAPH_INSTALL_LIB} COMPONENT ngraph)

if (NGRAPH_EXPORT_TARGETS_ENABLE)
    export(TARGETS ${TARGET_NAME} NAMESPACE ngraph:: APPEND FILE "${NGRAPH_TARGETS_FILE}")
endif()
