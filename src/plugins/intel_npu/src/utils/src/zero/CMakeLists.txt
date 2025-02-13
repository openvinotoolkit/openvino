# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME openvino_npu_zero_utils)

file(GLOB_RECURSE SOURCES *.cpp)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

add_library(${TARGET_NAME} STATIC ${SOURCES})
set_target_properties(
  ${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})
ov_add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})

add_library(openvino::npu_zero_utils ALIAS ${TARGET_NAME})
set_target_properties(${TARGET_NAME} PROPERTIES EXPORT_NAME npu_zero_utils)

add_library(level-zero-headers INTERFACE)
set_property(TARGET level-zero-headers APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES $<TARGET_PROPERTY:LevelZero::LevelZero,INTERFACE_INCLUDE_DIRECTORIES>)
add_library(LevelZero::Headers ALIAS level-zero-headers)

target_include_directories(${TARGET_NAME}
  PUBLIC
    $<TARGET_PROPERTY:LevelZero::Headers,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:LevelZero::NPUExt,INTERFACE_INCLUDE_DIRECTORIES>
    $<BUILD_INTERFACE:${NPU_UTILS_SOURCE_DIR}/include>)

target_link_libraries(${TARGET_NAME} PUBLIC openvino::runtime::dev)

#
# targets install
#

ov_install_static_lib(${TARGET_NAME} ${NPU_PLUGIN_COMPONENT})

ov_developer_package_export_targets(TARGET openvino::npu_zero_utils
                                    INSTALL_INCLUDE_DIRECTORIES
                                        $<BUILD_INTERFACE:${NPU_UTILS_SOURCE_DIR}/include>)

ov_developer_package_export_targets(TARGET level-zero-ext)
ov_install_static_lib(level-zero-ext ${NPU_PLUGIN_COMPONENT})

ov_developer_package_export_targets(TARGET level-zero-headers)
ov_install_static_lib(level-zero-headers ${NPU_PLUGIN_COMPONENT})

if(TARGET ze_loader)
    ov_developer_package_export_targets(TARGET ze_loader)
    ov_install_static_lib(ze_loader ${NPU_PLUGIN_COMPONENT})
    
    add_dependencies(${TARGET_NAME} ze_loader)
endif()
