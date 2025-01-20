# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "openvino_onnx_common")
set(OV_ONNX_NAMESPACE openvino_onnx)


file(GLOB_RECURSE LIBRARY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE PUBLIC_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp)

if(NOT ONNX_FOUND)
    message (STATUS "ONNX DBG Generating own onnx headers")
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

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj
source_group("src" FILES ${LIBRARY_SRC})
source_group("include" FILES ${PUBLIC_HEADERS} ${PUBLIC_HEADERS})

# Create static library
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
    message (STATUS "ONNX DBG Linked system onnx_proto and onnx")
    ov_link_system_libraries(${TARGET_NAME} PUBLIC onnx_proto onnx)
else()
    set(onnx_pb_file ${ONNX_COMMON_INCLUDE_DIR}/onnx/onnx_pb.h)
    file(WRITE ${onnx_pb_file} "// Copyright (C) 2018-2025 Intel Corporation\n// SPDX-License-Identifier: Apache-2.0\n// Auto generated file, DO NOT EDIT INLINE\n\n")
    file(APPEND ${onnx_pb_file} "#include \"onnx/onnx.pb.h\"\n")

    target_compile_definitions(${TARGET_NAME} PUBLIC ONNX_NAMESPACE=${OV_ONNX_NAMESPACE})
    if(ENABLE_SYSTEM_PROTOBUF)
        # use imported target name with namespace
        set(protobuf_target_name "protobuf::${protobuf_target_name}")
    else()
        set(protobuf_target_name libprotobuf)
    endif()

    message (STATUS "ONNX DBG Linked system Protobuf: " ${protobuf_target_name})
    ov_link_system_libraries(${TARGET_NAME} PUBLIC ${protobuf_target_name})

    # Files produced by protobuf may contain missing declarations
    if(CMAKE_COMPILER_IS_GNUCXX OR OV_COMPILER_IS_CLANG OR (OV_COMPILER_IS_INTEL_LLVM AND UNIX))
        target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wno-missing-declarations>)
    endif()
    # Protobuf generates files which cause a following warnings
    if(SUGGEST_OVERRIDE_SUPPORTED)
        target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wno-suggest-override>)
    endif()
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        # 4244 conversion from 'XXX' to 'YYY', possible loss of data
        ov_add_compiler_flags(/wd4244)
        # '<': signed/unsigned mismatch
        ov_add_compiler_flags(/wd4018)
    endif()
endif()

ov_add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})

ov_install_static_lib(${TARGET_NAME} ${OV_CPACK_COMP_CORE})
