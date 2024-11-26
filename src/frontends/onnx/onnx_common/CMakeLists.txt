# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "openvino_onnx_common")
set(OV_ONNX_NAMESPACE openvino_onnx)


file(GLOB_RECURSE LIBRARY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE PUBLIC_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp)

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj

source_group("src" FILES ${LIBRARY_SRC})
source_group("include" FILES ${PUBLIC_HEADERS} ${PUBLIC_HEADERS})

# Create static library

if(NOT ONNX_FOUND)
    # Generate protobuf file on build time for each '.proto' file in src/proto
    set(protofiles_root_dir "${CMAKE_CURRENT_SOURCE_DIR}/src/proto")
    file(GLOB_RECURSE proto_files ${protofiles_root_dir}/*.proto)

    foreach(proto_file IN LISTS proto_files)
        file(RELATIVE_PATH proto_file_relative "${CMAKE_SOURCE_DIR}" "${proto_file}")
        get_filename_component(FILE_WE ${proto_file} NAME_WE)
        file(RELATIVE_PATH relative_path ${protofiles_root_dir} ${proto_file})
        get_filename_component(relative_path ${relative_path} DIRECTORY)
        set(OUTPUT_PB_SRC ${CMAKE_CURRENT_BINARY_DIR}/${relative_path}/${FILE_WE}.pb.cc)
        set(OUTPUT_PB_HEADER ${CMAKE_CURRENT_BINARY_DIR}/${relative_path}/${FILE_WE}.pb.h)
        add_custom_command(
                OUTPUT "${OUTPUT_PB_SRC}" "${OUTPUT_PB_HEADER}"
                COMMAND ${PROTOC_EXECUTABLE} ARGS --cpp_out ${CMAKE_CURRENT_BINARY_DIR} -I ${protofiles_root_dir} ${proto_file}
                DEPENDS ${PROTOC_DEPENDENCY} ${proto_file}
                COMMENT "Running C++ protocol buffer compiler (${PROTOC_EXECUTABLE}) on ${proto_file_relative}"
                VERBATIM
                COMMAND_EXPAND_LISTS)
        list(APPEND LIBRARY_SRC "${OUTPUT_PB_SRC}")
        list(APPEND PUBLIC_HEADERS "${OUTPUT_PB_HEADER}")
    endforeach()
endif()

add_library(${TARGET_NAME} STATIC ${LIBRARY_SRC} ${PUBLIC_HEADERS})

set_target_properties(${TARGET_NAME} PROPERTIES
    INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

set(ONNX_COMMON_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(ONNX_COMMON_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_include_directories(${TARGET_NAME}
    PUBLIC $<BUILD_INTERFACE:${ONNX_COMMON_INCLUDE_DIR}>
           $<INSTALL_INTERFACE:${FRONTEND_INSTALL_INCLUDE}>
           $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    PRIVATE ${ONNX_COMMON_SRC_DIR})

target_link_libraries(${TARGET_NAME} PRIVATE openvino::runtime openvino::util)

if(ONNX_FOUND)
    ov_link_system_libraries(${TARGET_NAME} PUBLIC onnx_proto onnx)
else()
    target_compile_definitions(${TARGET_NAME} PUBLIC ONNX_NAMESPACE=${OV_ONNX_NAMESPACE})
    if(ENABLE_SYSTEM_PROTOBUF)
        # use imported target name with namespace
        set(protobuf_target_name "protobuf::${protobuf_target_name}")
    else()
        set(protobuf_target_name libprotobuf)
    endif()

    ov_link_system_libraries(${TARGET_NAME} PUBLIC ${protobuf_target_name})
endif()

ov_add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})

ov_add_compiler_flags(-Wno-missing-declarations)

ov_install_static_lib(${TARGET_NAME} ${OV_CPACK_COMP_CORE})
