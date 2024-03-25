# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(ENABLE_LTO)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endif()

set(TARGET_NAME "npu_level_zero_backend")
set(OBJ_TARGET_NAME "${TARGET_NAME}_obj")

file(GLOB_RECURSE SOURCES *.cpp *.hpp *.h)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

if(BUILD_SHARED_LIBS)
    add_library(${OBJ_TARGET_NAME} OBJECT ${SOURCES})
else()
    add_library(${OBJ_TARGET_NAME} STATIC ${SOURCES})
endif()

ov_add_clang_format_target(${OBJ_TARGET_NAME}_clang FOR_TARGETS ${OBJ_TARGET_NAME})

target_compile_definitions(${OBJ_TARGET_NAME}
    PRIVATE
        IMPLEMENT_OPENVINO_RUNTIME_PLUGIN)

target_include_directories(${OBJ_TARGET_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(${OBJ_TARGET_NAME}
    PRIVATE
        npu_al
        npu_zero_result_parser
        ze_loader
)

if(BUILD_SHARED_LIBS)
    add_library(${TARGET_NAME} STATIC)
    target_link_libraries(${TARGET_NAME} PRIVATE ${OBJ_TARGET_NAME})
else()
    add_library(${TARGET_NAME} ALIAS ${OBJ_TARGET_NAME})
endif()

set(NPU_LEVEL_ZERO_BACKEND_OBJ_TARGET "${OBJ_TARGET_NAME}" PARENT_SCOPE)

#
# targets install
#
if(NOT BUILD_SHARED_LIBS)
    install(TARGETS ${OBJ_TARGET_NAME} EXPORT OpenVINOTargets
            ARCHIVE DESTINATION ${OV_CPACK_ARCHIVEDIR} COMPONENT ${NPU_PLUGIN_COMPONENT}
            LIBRARY DESTINATION ${OV_CPACK_RUNTIMEDIR} COMPONENT ${NPU_PLUGIN_COMPONENT})
endif()
