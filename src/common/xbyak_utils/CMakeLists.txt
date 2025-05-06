# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME openvino_xbyak_utils)

set(XBYAK_UTILS_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/)
file(GLOB_RECURSE PUBLIC_HEADERS ${XBYAK_UTILS_INCLUDE_DIR}/*.hpp)
file(GLOB_RECURSE LIBRARY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj

source_group("src" FILES ${LIBRARY_SRC})
source_group("include" FILES ${PUBLIC_HEADERS})

# Create interface library
add_library(${TARGET_NAME} STATIC ${LIBRARY_SRC} ${PUBLIC_HEADERS})

add_library(openvino::xbyak_utils ALIAS ${TARGET_NAME})
set_target_properties(${TARGET_NAME} PROPERTIES EXPORT_NAME xbyak_utils)

target_include_directories(${TARGET_NAME}
    PUBLIC $<BUILD_INTERFACE:${XBYAK_UTILS_INCLUDE_DIR}>
    PRIVATE $<BUILD_INTERFACE:${OpenVINO_SOURCE_DIR}/src/core/include>)

target_include_directories(${TARGET_NAME} SYSTEM PUBLIC
    $<BUILD_INTERFACE:$<$<TARGET_EXISTS:xbyak::xbyak>:$<TARGET_PROPERTY:xbyak::xbyak,INTERFACE_INCLUDE_DIRECTORIES>>>)

target_compile_definitions(${TARGET_NAME} PUBLIC XBYAK_NO_OP_NAMES XBYAK64 PRIVATE IMPLEMENT_OPENVINO_API)

ov_add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})

# install & export

ov_install_static_lib(${TARGET_NAME} ${OV_CPACK_COMP_CORE})

ov_developer_package_export_targets(TARGET openvino::xbyak_utils
                                    INSTALL_INCLUDE_DIRECTORIES "${XBYAK_UTILS_INCLUDE_DIR}/")