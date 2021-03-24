# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME itt)

file(GLOB_RECURSE SOURCES "src/*.cpp" "src/*.hpp")

add_library(${TARGET_NAME} STATIC ${SOURCES})

add_library(openvino::itt ALIAS ${TARGET_NAME})

target_link_libraries(${TARGET_NAME} PUBLIC openvino::pp)

if(TARGET ittnotify)
    target_link_libraries(${TARGET_NAME} PUBLIC ittnotify)
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(${TARGET_NAME} PRIVATE
        "-Wall"
    )
endif()

target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_cpplint_target(${TARGET_NAME}_cpplint FOR_TARGETS ${TARGET_NAME})
