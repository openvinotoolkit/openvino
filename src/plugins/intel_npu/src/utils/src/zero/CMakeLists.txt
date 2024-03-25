# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(ENABLE_LTO)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endif()

set(TARGET_NAME npu_zero_result_parser)

file(GLOB_RECURSE SOURCES *.cpp)

add_library(${TARGET_NAME} STATIC ${SOURCES})
ov_add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})

target_include_directories(${TARGET_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${NPU_PLUGIN_SOURCE_DIR}/thirdparty/level-zero/include>
        $<BUILD_INTERFACE:${NPU_PLUGIN_SOURCE_DIR}/thirdparty/level-zero-ext>
        $<BUILD_INTERFACE:${NPU_UTILS_SOURCE_DIR}/include>
)

#
# targets install
#
if (NOT BUILD_SHARED_LIBS)
    install(TARGETS ${TARGET_NAME} EXPORT OpenVINOTargets
            RUNTIME DESTINATION ${OV_CPACK_RUNTIMEDIR} COMPONENT ${NPU_PLUGIN_COMPONENT}
            ARCHIVE DESTINATION ${OV_CPACK_ARCHIVEDIR} COMPONENT ${NPU_PLUGIN_COMPONENT}
            LIBRARY DESTINATION ${OV_CPACK_LIBRARYDIR} COMPONENT ${NPU_PLUGIN_COMPONENT})
endif()
