# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(ENABLE_LTO)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endif()

set(TARGET_NAME npu_al)

file(GLOB_RECURSE SOURCES *.cpp *.hpp *.h)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

add_library(${TARGET_NAME} STATIC ${SOURCES})

target_include_directories(${TARGET_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

if(ENABLE_MLIR_COMPILER)
    target_compile_definitions(${TARGET_NAME} PRIVATE ENABLE_MLIR_COMPILER)
endif()

target_link_libraries(${TARGET_NAME}
    PUBLIC
        npu_logger_utils
        openvino::runtime::dev
)

ov_add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})

#
# targets install
#

if (NOT BUILD_SHARED_LIBS)
    install(TARGETS ${TARGET_NAME} EXPORT OpenVINOTargets
            ARCHIVE DESTINATION ${OV_CPACK_ARCHIVEDIR} COMPONENT ${NPU_PLUGIN_COMPONENT})
endif()

ov_developer_package_export_targets(TARGET npu_al
                                    INSTALL_INCLUDE_DIRECTORIES
                                        ${CMAKE_CURRENT_SOURCE_DIR}/include)
