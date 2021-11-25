# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_util)

file(GLOB_RECURSE LIBRARY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE PUBLIC_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp)

set(UTIL_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/)

if (WIN32)
    # Remove linux specific files
    file(GLOB_RECURSE LIN_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/os/lin/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/os/lin/*.hpp)
    list(REMOVE_ITEM LIBRARY_SRC ${LIN_FILES})
else()
    # Remove windows specific files
    file(GLOB_RECURSE WIN_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/os/win/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/os/win/*.hpp)
    list(REMOVE_ITEM LIBRARY_SRC ${WIN_FILES})
endif()

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj

source_group("src" FILES ${LIBRARY_SRC})
source_group("include" FILES ${PUBLIC_HEADERS})

# Create shared library
add_library(${TARGET_NAME}_obj OBJECT ${LIBRARY_SRC} ${PUBLIC_HEADERS})
target_include_directories(${TARGET_NAME}_obj PUBLIC $<BUILD_INTERFACE:${UTIL_INCLUDE_DIR}>)

add_library(${TARGET_NAME} INTERFACE)
add_library(openvino::util ALIAS ${TARGET_NAME})

target_include_directories(${TARGET_NAME} PUBLIC INTERFACE $<BUILD_INTERFACE:${UTIL_INCLUDE_DIR}>)
target_link_libraries(${TARGET_NAME} PRIVATE INTERFACE openvino::runtime)


# ov_install_static_lib(${TARGET_NAME} ngraph)

add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME}_obj)
ov_ncc_naming_style(FOR_TARGET ${TARGET_NAME}_obj
                    INCLUDE_DIRECTORY ${UTIL_INCLUDE_DIR})

openvino_developer_export_targets(COMPONENT openvino_common TARGETS ${TARGET_NAME})
