# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME openvino_util)

set(UTIL_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/)

file(GLOB_RECURSE LIBRARY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE PUBLIC_HEADERS ${UTIL_INCLUDE_DIR}/*.hpp)

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

set_source_files_properties(
    "${CMAKE_CURRENT_SOURCE_DIR}/src/file_util.cpp"
    PROPERTIES COMPILE_DEFINITIONS OpenVINO_VERSION="${OpenVINO_VERSION}")

source_group("src" FILES ${LIBRARY_SRC})
source_group("include" FILES ${PUBLIC_HEADERS})

# Create shared library
add_library(${TARGET_NAME} STATIC ${LIBRARY_SRC} ${PUBLIC_HEADERS})

add_library(openvino::util ALIAS ${TARGET_NAME})
set_target_properties(${TARGET_NAME} PROPERTIES EXPORT_NAME util)

target_link_libraries(${TARGET_NAME}
    PRIVATE ${CMAKE_DL_LIBS}
    PUBLIC openvino::pugixml
           $<$<AND:$<CXX_COMPILER_ID:GNU>,$<VERSION_LESS:$<CXX_COMPILER_VERSION>,9.1>>:stdc++fs>
           $<$<AND:$<CXX_COMPILER_ID:Clang>,$<VERSION_LESS:$<CXX_COMPILER_VERSION>,9.0>>:c++fs>)

if (WIN32)
    target_link_libraries(${TARGET_NAME} PRIVATE Shlwapi)
endif()

target_include_directories(${TARGET_NAME} PUBLIC $<BUILD_INTERFACE:${UTIL_INCLUDE_DIR}>)

ov_add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})
ov_ncc_naming_style(FOR_TARGET ${TARGET_NAME}
                    SOURCE_DIRECTORIES ${UTIL_INCLUDE_DIR})

set_target_properties(${TARGET_NAME} PROPERTIES PCH_CORE_HEADER_PATH "${UTIL_INCLUDE_DIR}/openvino/pch/precomp_std.hpp")

ov_build_target_faster(${TARGET_NAME} PCH)

# install & export

ov_install_static_lib(${TARGET_NAME} ${OV_CPACK_COMP_CORE})

ov_developer_package_export_targets(TARGET ${TARGET_NAME}
                                    INSTALL_INCLUDE_DIRECTORIES "${UTIL_INCLUDE_DIR}/")
