# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "openvino_xml_util")

set(TARGET_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

file(GLOB_RECURSE LIBRARY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE PUBLIC_HEADERS ${TARGET_INCLUDE_DIR}/*.hpp)

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj

source_group("src" FILES ${LIBRARY_SRC})
source_group("include" FILES ${PUBLIC_HEADERS})

# Create static library
add_library(${TARGET_NAME} STATIC ${LIBRARY_SRC} ${PUBLIC_HEADERS})

add_library(openvino::xml_util ALIAS ${TARGET_NAME})
set_target_properties(${TARGET_NAME} PROPERTIES EXPORT_NAME openvino_xml_util)

target_include_directories(${TARGET_NAME} PUBLIC
    $<BUILD_INTERFACE:${TARGET_INCLUDE_DIR}>
    $<TARGET_PROPERTY:openvino::core::dev,INTERFACE_INCLUDE_DIRECTORIES>)

ov_add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})
target_link_libraries(${TARGET_NAME} PRIVATE openvino::runtime)

if(NOT BUILD_SHARED_LIBS)
    target_compile_definitions(${TARGET_NAME} PUBLIC OPENVINO_STATIC_LIBRARY)
endif()

ov_build_target_faster(${TARGET_NAME} PCH)

# developer package

ov_install_static_lib(${TARGET_NAME} ${OV_CPACK_COMP_CORE})

ov_developer_package_export_targets(TARGET ${TARGET_NAME}
                                    INSTALL_INCLUDE_DIRECTORIES "${TARGET_INCLUDE_DIR}/")
