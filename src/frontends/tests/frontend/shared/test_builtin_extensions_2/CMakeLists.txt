# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "test_builtin_extensions_2")

file(GLOB_RECURSE LIBRARY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB_RECURSE LIBRARY_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

# Create library
add_library(${TARGET_NAME} SHARED ${LIBRARY_SRC} ${LIBRARY_HEADERS})

target_link_libraries(${TARGET_NAME} PRIVATE openvino::runtime::dev offline_transformations)

add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})

if(WIN32)
    install(TARGETS ${TARGET_NAME}
            RUNTIME DESTINATION tests COMPONENT tests EXCLUDE_FROM_ALL
            LIBRARY DESTINATION tests COMPONENT tests EXCLUDE_FROM_ALL)
else()
    install(TARGETS ${TARGET_NAME}
            RUNTIME DESTINATION tests/lib COMPONENT tests EXCLUDE_FROM_ALL
            LIBRARY DESTINATION tests/lib COMPONENT tests EXCLUDE_FROM_ALL)
endif()
