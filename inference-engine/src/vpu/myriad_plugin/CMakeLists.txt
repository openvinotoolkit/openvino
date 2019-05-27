# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "myriadPlugin")

file(GLOB_RECURSE SOURCES *.cpp *.hpp *.h)

addVersionDefines(myriad_plugin.cpp CI_BUILD_NUMBER)

add_library(${TARGET_NAME} SHARED ${SOURCES})

target_include_directories(${TARGET_NAME}
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
    SYSTEM PRIVATE
        "${IE_MAIN_SOURCE_DIR}/include"
        "${IE_MAIN_SOURCE_DIR}/src/inference_engine")

target_compile_definitions(${TARGET_NAME} PRIVATE IMPLEMENT_INFERENCE_ENGINE_PLUGIN)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(${TARGET_NAME}
        PRIVATE
            -Wall)
endif()

if (ENABLE_MYX_PCIE)
    target_compile_definitions(${TARGET_NAME} PRIVATE USE_PCIE)
elseif (ENABLE_MYRIAD_NO_BOOT)
    target_compile_definitions(${TARGET_NAME} PRIVATE NO_BOOT)
endif()

target_link_libraries(${TARGET_NAME}
    PRIVATE
        ${INTEL_ITT_LIBS} inference_engine vpu_graph_transformer mvnc)

set_target_properties(${TARGET_NAME} PROPERTIES
    COMPILE_PDB_NAME ${TARGET_NAME})

add_cpplint_target(${TARGET_NAME}_cpplint FOR_TARGETS ${TARGET_NAME})
