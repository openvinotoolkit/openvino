# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "paddlepaddle_ngraph_frontend")

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
add_library(${TARGET_NAME} ${LIBRARY_SRC} ${LIBRARY_HEADERS} ${LIBRARY_PUBLIC_HEADERS} ${PROTO_SRCS} ${PROTO_HDRS})
add_library(openvino::frontend::paddlepaddle ALIAS ${TARGET_NAME})

add_dependencies(${TARGET_NAME} paddlepaddle_ngraph_frontend_proto)

ov_ncc_naming_style(FOR_TARGET ${TARGET_NAME}
                    INCLUDE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include"
                    ADDITIONAL_INCLUDE_DIRECTORIES
                        $<TARGET_PROPERTY:frontend_manager::static,INTERFACE_INCLUDE_DIRECTORIES>)

target_include_directories(${TARGET_NAME}
        PUBLIC
            $<BUILD_INTERFACE:${${TARGET_NAME}_INCLUDE_DIR}>
            $<INSTALL_INTERFACE:${FRONTEND_INSTALL_INCLUDE}>
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_BINARY_DIR})

target_include_directories(${TARGET_NAME} SYSTEM PRIVATE ${Protobuf_INCLUDE_DIRS}
                                                         ${CMAKE_CURRENT_BINARY_DIR})

if(COMMAND ie_add_vs_version_file)
    ie_add_vs_version_file(NAME ${TARGET_NAME}
                           FILEDESCRIPTION "FrontEnd to load and convert PaddlePaddle file format")
endif()

link_system_libraries(${TARGET_NAME} PRIVATE ${Protobuf_LITE_LIBRARIES})

target_link_libraries(${TARGET_NAME} PRIVATE frontend_manager::static
                                     PRIVATE ngraph::builder inference_engine_transformations)

add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME}
                        EXCLUDE_PATTERNS ${PROTO_SRCS} ${PROTO_HDRS})

set_target_properties(${TARGET_NAME} PROPERTIES EXPORT_NAME frontend::paddlepaddle)
install(TARGETS ${TARGET_NAME} EXPORT OpenVINOTargets
        RUNTIME DESTINATION ${IE_CPACK_RUNTIME_PATH} COMPONENT ngraph
        ARCHIVE DESTINATION ${IE_CPACK_ARCHIVE_PATH} COMPONENT ngraph
        LIBRARY DESTINATION ${IE_CPACK_LIBRARY_PATH} COMPONENT ngraph)

install(DIRECTORY ${${TARGET_NAME}_INCLUDE_DIR}/paddlepaddle_frontend
        DESTINATION ${FRONTEND_INSTALL_INCLUDE}
        COMPONENT ngraph_dev
        FILES_MATCHING PATTERN "*.hpp")

export(TARGETS ${TARGET_NAME} NAMESPACE openvino::
       APPEND FILE "${CMAKE_BINARY_DIR}/OpenVINOTargets.cmake")
