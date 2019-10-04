# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "myriadPlugin")

file(GLOB_RECURSE SOURCES *.cpp *.hpp *.h)

ie_add_plugin(NAME ${TARGET_NAME}
              DEVICE_NAME "MYRIAD"
              SOURCES ${SOURCES}
              VERSION_DEFINES_FOR api/myriad_api.cpp)

add_dependencies(${TARGET_NAME} vpu_copy_firmware)
if(TARGET vpu_compile_custom_kernels)
    add_dependencies(${TARGET_NAME} vpu_compile_custom_kernels)
endif()

target_include_directories(${TARGET_NAME}
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
    SYSTEM PRIVATE
        "${IE_MAIN_SOURCE_DIR}/include"
        "${IE_MAIN_SOURCE_DIR}/src/inference_engine")

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(${TARGET_NAME}
        PRIVATE
            -Wall)
endif()

if (ENABLE_MYRIAD_NO_BOOT)
    target_compile_definitions(${TARGET_NAME} PRIVATE NO_BOOT)
endif()

target_link_libraries(${TARGET_NAME}
    PRIVATE
        ${INTEL_ITT_LIBS} inference_engine vpu_graph_transformer mvnc)
