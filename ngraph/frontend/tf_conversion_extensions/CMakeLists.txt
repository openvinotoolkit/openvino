# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "tf_conversion_extensions")

file(GLOB_RECURSE LIBRARY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE LIBRARY_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp)
file(GLOB_RECURSE LIBRARY_PUBLIC_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp)

set(${TARGET_NAME}_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

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
            COMMAND ${PROTOC_EXECUTABLE} ARGS --cpp_out ${CMAKE_CURRENT_BINARY_DIR} -I ${FILE_DIR} ${FILE_WE}.proto
            DEPENDS ${PROTOC_EXECUTABLE} ${GENERATED_PROTO}
            COMMENT "Running C++ protocol buffer compiler (${PROTOC_EXECUTABLE}) on ${GENERATED_PROTO}"
            VERBATIM
            COMMAND_EXPAND_LISTS)
    list(APPEND PROTO_SRCS "${OUTPUT_PB_SRC}")
    list(APPEND PROTO_HDRS "${OUTPUT_PB_HEADER}")
endforeach()

add_custom_target(${TARGET_NAME}_proto DEPENDS ${PROTO_SRCS} ${PROTO_HDRS})

set_source_files_properties(${PROTO_SRCS} ${PROTO_HDRS} PROPERTIES GENERATED TRUE)

# Disable all warnings for generated code
set_source_files_properties(${PROTO_SRCS} ${PROTO_HDRS} PROPERTIES COMPILE_OPTIONS -w)

# Create library
add_library(${TARGET_NAME} SHARED ${LIBRARY_SRC} ${LIBRARY_HEADERS} ${LIBRARY_PUBLIC_HEADERS} ${PROTO_SRCS} ${PROTO_HDRS})

get_target_property(TENSORFLOW_FRONTEND_SRC_DIR tensorflow_ov_frontend SOURCE_DIR)
target_include_directories(${TARGET_NAME}
        PUBLIC
            $<BUILD_INTERFACE:${${TARGET_NAME}_INCLUDE_DIR}>
            $<INSTALL_INTERFACE:${FRONTEND_INSTALL_INCLUDE}>
        PRIVATE
            ${TENSORFLOW_FRONTEND_SRC_DIR}/src/
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_BINARY_DIR})

target_link_libraries(${TARGET_NAME} PRIVATE frontend_manager::static openvino::util tensorflow_ov_frontend libprotobuf)

add_dependencies(${TARGET_NAME} tensorflow_ov_frontend)

install(TARGETS ${TARGET_NAME}
        RUNTIME DESTINATION ${IE_CPACK_RUNTIME_PATH} COMPONENT tests EXCLUDE_FROM_ALL
        ARCHIVE DESTINATION ${IE_CPACK_ARCHIVE_PATH} COMPONENT tests EXCLUDE_FROM_ALL
        LIBRARY DESTINATION ${IE_CPACK_LIBRARY_PATH} COMPONENT tests EXCLUDE_FROM_ALL)
