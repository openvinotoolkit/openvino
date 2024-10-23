# Copyright (C) 2018-2024 Intel Corporation
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

target_include_directories(${TARGET_NAME}
  PUBLIC
    $<TARGET_PROPERTY:LevelZero::LevelZero,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:LevelZero::NPUExt,INTERFACE_INCLUDE_DIRECTORIES>
    $<BUILD_INTERFACE:${NPU_UTILS_SOURCE_DIR}/include>)


target_link_libraries(
  ${TARGET_NAME}
  PRIVATE openvino::runtime openvino::runtime::dev ze_loader)

#
# targets install
#
ov_install_static_lib(${TARGET_NAME} ${NPU_INTERNAL_COMPONENT})

ov_developer_package_export_targets(TARGET openvino::npu_zero_utils
                                    INSTALL_INCLUDE_DIRECTORIES
                                        $<BUILD_INTERFACE:${NPU_UTILS_SOURCE_DIR}/include>)

if(TARGET ze_loader)
    install(TARGETS ze_loader EXPORT OpenVINOTargets)

    install(TARGETS utils EXPORT OpenVINOTargets)

    install(TARGETS level-zero-ext EXPORT OpenVINOTargets)

    export(TARGETS ze_loader NAMESPACE openvino::
        APPEND FILE "${CMAKE_BINARY_DIR}/OpenVINOTargets.cmake")

    export(TARGETS utils NAMESPACE openvino::
        APPEND FILE "${CMAKE_BINARY_DIR}/OpenVINOTargets.cmake")

    export(TARGETS level-zero-ext NAMESPACE openvino::
        APPEND FILE "${CMAKE_BINARY_DIR}/OpenVINOTargets.cmake")

    # Support tests to run with ze_loader
    install(TARGETS ze_loader
        LIBRARY DESTINATION tests COMPONENT tests EXCLUDE_FROM_ALL)
endif()
