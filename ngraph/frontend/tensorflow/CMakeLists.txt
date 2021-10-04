# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "tensorflow_ngraph_frontend")

if(SUGGEST_OVERRIDE_SUPPORTED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-suggest-override")
endif()
if(CMAKE_COMPILER_IS_GNUCXX)
    ie_add_compiler_flags(-Wno-undef)
elseif(UNIX AND CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -diag-disable=10121")
endif()

file(GLOB_RECURSE LIBRARY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE LIBRARY_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp)
file(GLOB_RECURSE LIBRARY_PUBLIC_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp)

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

if(CMAKE_CROSSCOMPILING)
    set(TENSORFLOW_PROTOC_EXECUTABLE ${SYSTEM_PROTOC})
else()
    set(TENSORFLOW_PROTOC_EXECUTABLE $<TARGET_FILE:protoc>)
    #set(TENSORFLOW_PROTOC_EXECUTABLE ${SYSTEM_PROTOC})
endif()

message("TENSORFLOW_PROTOC_EXECUTABLE is [${TENSORFLOW_PROTOC_EXECUTABLE}]")

foreach(INFILE ${proto_files})
    get_filename_component(FILE_DIR ${INFILE} DIRECTORY)
    get_filename_component(FILE_WE ${INFILE} NAME_WE)
    set(OUTPUT_PB_SRC ${CMAKE_CURRENT_BINARY_DIR}/${FILE_WE}.pb.cc)
    set(OUTPUT_PB_HEADER ${CMAKE_CURRENT_BINARY_DIR}/${FILE_WE}.pb.h)
    set(GENERATED_PROTO ${INFILE})
    add_custom_command(
            OUTPUT "${OUTPUT_PB_SRC}" "${OUTPUT_PB_HEADER}"
            COMMAND ${TENSORFLOW_PROTOC_EXECUTABLE} ARGS --cpp_out ${CMAKE_CURRENT_BINARY_DIR} -I ${FILE_DIR} ${FILE_WE}.proto
            DEPENDS ${TENSORFLOW_PROTOC_EXECUTABLE} ${GENERATED_PROTO}
            COMMENT "Running C++ protocol buffer compiler (${TENSORFLOW_PROTOC_EXECUTABLE}) on ${GENERATED_PROTO}"
            VERBATIM)
    list(APPEND PROTO_SRCS "${OUTPUT_PB_SRC}")
    list(APPEND PROTO_HDRS "${OUTPUT_PB_HEADER}")
endforeach()

add_custom_target(${TARGET_NAME}_proto DEPENDS ${PROTO_SRCS} ${PROTO_HDRS})

set_source_files_properties(${PROTO_SRCS} ${PROTO_HDRS} PROPERTIES GENERATED TRUE)

# Disable all warnings for generated code
set_source_files_properties(${PROTO_SRCS} ${PROTO_HDRS} PROPERTIES COMPILE_OPTIONS -w)

# Create shared library
add_library(${TARGET_NAME} SHARED  ${LIBRARY_SRC} ${LIBRARY_HEADERS} ${LIBRARY_PUBLIC_HEADERS} ${PROTO_SRCS} ${PROTO_HDRS})

add_dependencies(${TARGET_NAME} libprotobuf tensorflow_ngraph_frontend_proto)

target_include_directories(${TARGET_NAME}
        PUBLIC
        $<BUILD_INTERFACE:${${TARGET_NAME}_INCLUDE_DIR}>
        $<INSTALL_INTERFACE:${FRONTEND_INSTALL_INCLUDE}>
        PRIVATE
        ${Protobuf_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_BINARY_DIR})

if(COMMAND ie_add_vs_version_file)
    ie_add_vs_version_file(NAME ${TARGET_NAME}
            FILEDESCRIPTION "FrontEnd to load and convert TensorFlow file format")
endif()

target_link_libraries(${TARGET_NAME} PUBLIC frontend_manager
        PRIVATE ngraph::builder inference_engine inference_engine_transformations libprotobuf openvino::util)

add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME}
        EXCLUDE_PATTERNS ${PROTO_SRCS} ${PROTO_HDRS})

install(TARGETS ${TARGET_NAME} EXPORT ngraphTargets
        RUNTIME DESTINATION ${NGRAPH_INSTALL_LIB} COMPONENT ngraph
        ARCHIVE DESTINATION ${NGRAPH_INSTALL_LIB} COMPONENT ngraph
        LIBRARY DESTINATION ${NGRAPH_INSTALL_LIB} COMPONENT ngraph)

install(DIRECTORY ${${TARGET_NAME}_INCLUDE_DIR}/tensorflow_frontend
        DESTINATION ${FRONTEND_INSTALL_INCLUDE}
        COMPONENT ngraph_dev
        FILES_MATCHING PATTERN "*.hpp"
        )

export(TARGETS ${TARGET_NAME} NAMESPACE ngraph:: APPEND FILE "${NGRAPH_TARGETS_FILE}")
