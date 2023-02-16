# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME util)

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
add_library(${TARGET_NAME} STATIC ${LIBRARY_SRC} ${PUBLIC_HEADERS})

add_library(openvino::util ALIAS ${TARGET_NAME})

target_link_libraries(${TARGET_NAME} PRIVATE ${CMAKE_DL_LIBS})
if (WIN32)
    target_link_libraries(${TARGET_NAME} PRIVATE Shlwapi)
endif()
target_include_directories(${TARGET_NAME} PUBLIC
    $<BUILD_INTERFACE:${UTIL_INCLUDE_DIR}>)

ov_install_static_lib(${TARGET_NAME} ${OV_CPACK_COMP_CORE})

add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})
ov_ncc_naming_style(FOR_TARGET ${TARGET_NAME}
                    SOURCE_DIRECTORY ${UTIL_INCLUDE_DIR})

openvino_developer_export_targets(COMPONENT core TARGETS ${TARGET_NAME})
