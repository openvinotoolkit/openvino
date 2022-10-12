# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "offline_transformations")

file(GLOB_RECURSE LIBRARY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE PUBLIC_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp)

set(PUBLIC_HEADERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj

source_group("src" FILES ${LIBRARY_SRC})
source_group("include" FILES ${PUBLIC_HEADERS})

# Create static library

add_library(${TARGET_NAME} STATIC ${LIBRARY_SRC} ${PUBLIC_HEADERS})

target_link_libraries(${TARGET_NAME} PRIVATE openvino::runtime::dev openvino::itt openvino::pugixml ngraph::reference
                                             openvino::runtime nlohmann_json_schema_validator)

set_source_files_properties(src/json_config.json_config.cpp INCLUDE_DIRECTORIES
    $<TARGET_PROPERTY:nlohmann_json_schema_validator,INTERFACE_INCLUDE_DIRECTORIES>)
set_source_files_properties(src/generate_mapping_file.cpp INCLUDE_DIRECTORIES
    $<TARGET_PROPERTY:openvino::pugixml,INTERFACE_INCLUDE_DIRECTORIES>)

target_include_directories(${TARGET_NAME} PUBLIC ${PUBLIC_HEADERS_DIR}
                                          PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")

add_cpplint_target(${TARGET_NAME}_cpplint FOR_TARGETS ${TARGET_NAME})

add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})

# developer package

openvino_developer_export_targets(COMPONENT core TARGETS ${TARGET_NAME})
