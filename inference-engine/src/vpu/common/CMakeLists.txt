# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "vpu_common_lib")

file(GLOB_RECURSE SOURCES *.cpp *.hpp *.h)

add_library(${TARGET_NAME} STATIC ${SOURCES})

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # TODO: enable some day and fix all warnings
    #     target_compile_options(${TARGET_NAME} PRIVATE "-Wall")
    target_compile_options(${TARGET_NAME} PRIVATE "-Werror=unused-variable")
    target_compile_options(${TARGET_NAME} PRIVATE "-Werror=unused-function")
    target_compile_options(${TARGET_NAME} PRIVATE "-Werror=strict-aliasing")
endif()

target_include_directories(${TARGET_NAME}
        PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include")

target_include_directories(${TARGET_NAME}
        SYSTEM PUBLIC
        "${IE_MAIN_SOURCE_DIR}/include"
        "${IE_MAIN_SOURCE_DIR}/src/inference_engine")

if(WIN32)
    target_compile_definitions(${TARGET_NAME} PRIVATE NOMINMAX)

    set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_PDB_NAME ${TARGET_NAME})
endif()

add_cpplint_target(${TARGET_NAME}_cpplint FOR_TARGETS ${TARGET_NAME})
add_cppcheck(${TARGET_NAME})

#
# developer package
#

export(TARGETS ${TARGET_NAME} NAMESPACE IE::
        APPEND FILE "${CMAKE_BINARY_DIR}/targets_developer.cmake")
