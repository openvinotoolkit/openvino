# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "openvino_tensorflow_common")
get_filename_component(root_dir "${CMAKE_CURRENT_SOURCE_DIR}" DIRECTORY)

file(GLOB_RECURSE LIBRARY_SRC ${root_dir}/src/*.cpp)
file(GLOB_RECURSE LIBRARY_HEADERS ${root_dir}/src/*.hpp ${root_dir}/dev_api/*.hpp)
file(GLOB_RECURSE LIBRARY_PUBLIC_HEADERS ${root_dir}/include/*.hpp)

add_library(${TARGET_NAME} STATIC ${LIBRARY_SRC} ${LIBRARY_HEADERS} ${LIBRARY_PUBLIC_HEADERS})
target_link_libraries(${TARGET_NAME} PRIVATE openvino::util)
add_library(openvino::frontend::tensorflow_common ALIAS ${TARGET_NAME})

add_library(${TARGET_NAME}_dev INTERFACE)
target_link_libraries(${TARGET_NAME}_dev INTERFACE ${TARGET_NAME})
target_include_directories(${TARGET_NAME}_dev INTERFACE $<BUILD_INTERFACE:${root_dir}/dev_api>)
add_library(openvino::frontend::tensorflow_common::dev ALIAS ${TARGET_NAME}_dev)

target_include_directories(${TARGET_NAME}
        PUBLIC
        $<BUILD_INTERFACE:${root_dir}/include>
        PRIVATE
        ${root_dir}/src
        ${root_dir}/dev_api
        $<TARGET_PROPERTY:openvino::runtime::dev,INTERFACE_INCLUDE_DIRECTORIES>)

add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})
ov_ncc_naming_style(FOR_TARGET ${TARGET_NAME}
        SOURCE_DIRECTORY "${root_dir}/include"
        ADDITIONAL_INCLUDE_DIRECTORIES
        $<TARGET_PROPERTY:frontend_common::static,INTERFACE_INCLUDE_DIRECTORIES>)
ov_install_static_lib(${TARGET_NAME} ${OV_CPACK_COMP_CORE})
ov_install_static_lib(${TARGET_NAME}_dev ${OV_CPACK_COMP_CORE})
