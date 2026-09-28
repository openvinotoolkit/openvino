# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "openvino_offline_transformations")

set(PUBLIC_HEADERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

set(LIBRARY_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/src/compress_quantize_weigths.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/pruning/init_const_mask.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/pruning/init_masks.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/pruning/mask_attribute.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/pruning/propagate_masks.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/pruning/pruning.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/pruning/shrink_weights.cpp
)

set(PUBLIC_HEADERS
        ${PUBLIC_HEADERS_DIR}/compress_quantize_weights.hpp
        ${PUBLIC_HEADERS_DIR}/mask_attribute.hpp
        ${PUBLIC_HEADERS_DIR}/pruning.hpp
)

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj
source_group("src" FILES ${LIBRARY_SRC})
source_group("include" FILES ${PUBLIC_HEADERS})

# Create static library
add_library(${TARGET_NAME} STATIC ${LIBRARY_SRC} ${PUBLIC_HEADERS})

add_library(openvino::offline_transformations ALIAS ${TARGET_NAME})
set_target_properties(${TARGET_NAME} PROPERTIES EXPORT_NAME offline_transformations)

target_link_libraries(${TARGET_NAME} PRIVATE openvino::core::dev openvino::reference openvino::runtime)

target_include_directories(${TARGET_NAME} PUBLIC $<BUILD_INTERFACE:${PUBLIC_HEADERS_DIR}>
                                          PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
                                                  $<TARGET_PROPERTY:openvino::pugixml,INTERFACE_INCLUDE_DIRECTORIES>)

ov_add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})

ov_build_target_faster(${TARGET_NAME} PCH)
# install & export

ov_developer_package_export_targets(TARGET ${TARGET_NAME}
                                    INSTALL_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/include/")
