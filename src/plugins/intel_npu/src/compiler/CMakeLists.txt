# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(ENABLE_LTO)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endif()

set(TARGET_NAME "npu_driver_compiler_adapter")

file(GLOB_RECURSE SOURCES *.cpp *.hpp *.h)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

add_library(${TARGET_NAME} STATIC ${SOURCES})

ov_build_target_faster(${TARGET_NAME}
    PCH PRIVATE "src/precomp.hpp"
)

ov_add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})

target_include_directories(${TARGET_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(${TARGET_NAME}
    PUBLIC
        npu_al
        ze_loader
    PRIVATE
        npu_zero_result_parser
        openvino::runtime
        openvino::runtime::dev
)

#
# targets install
#
if(NOT BUILD_SHARED_LIBS)
    install(TARGETS ${TARGET_NAME} EXPORT OpenVINOTargets
            ARCHIVE DESTINATION ${OV_CPACK_ARCHIVEDIR} COMPONENT ${NPU_PLUGIN_COMPONENT}
            LIBRARY DESTINATION ${OV_CPACK_RUNTIMEDIR} COMPONENT ${NPU_PLUGIN_COMPONENT})
endif()
