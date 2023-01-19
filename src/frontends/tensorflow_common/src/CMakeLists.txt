# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "openvino_tensorflow_common")
get_filename_component(root_dir "${CMAKE_CURRENT_SOURCE_DIR}" DIRECTORY)

file(GLOB_RECURSE LIBRARY_SRC ${root_dir}/src/*.cpp)
file(GLOB_RECURSE LIBRARY_HEADERS ${root_dir}/include/*.hpp)
#file(GLOB_RECURSE LIBRARY_PUBLIC_HEADERS ${root_dir}/include/*.hpp)
#
add_library(${TARGET_NAME} STATIC ${LIBRARY_SRC} ${LIBRARY_HEADERS})
target_link_libraries(${TARGET_NAME} PRIVATE openvino::util)
add_library(openvino::frontend::tensorflow_common ALIAS ${TARGET_NAME})

target_include_directories(${TARGET_NAME}
        PUBLIC
        ${root_dir}/include
        PRIVATE
        ${root_dir}/src
        $<TARGET_PROPERTY:openvino::runtime::dev,INTERFACE_INCLUDE_DIRECTORIES>)

add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})
ov_install_static_lib(${TARGET_NAME} ${OV_CPACK_COMP_CORE})
