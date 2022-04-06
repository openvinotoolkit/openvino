# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "frontend_shared_test_classes")

file(GLOB_RECURSE LIBRARY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE LIBRARY_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp)

add_library(${TARGET_NAME} STATIC EXCLUDE_FROM_ALL ${LIBRARY_SRC} ${LIBRARY_HEADERS})

add_subdirectory(test_builtin_extensions_1)
add_subdirectory(test_builtin_extensions_2)

target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../..)

target_link_libraries(${TARGET_NAME}
        PUBLIC
                engines_test_util offline_transformations
                commonTestUtils ngraph_test_util openvino::util
                openvino::runtime
        PRIVATE
                cnpy
                # TODO: replace with nlohmann_json::nlohmann_json
                # but LTO fails for some reason
                nlohmann_json_schema_validator
                test_builtin_extensions_1 test_builtin_extensions_2)

target_compile_definitions(${TARGET_NAME}
    PRIVATE
        SHARED_LIB_PREFIX="${CMAKE_SHARED_LIBRARY_PREFIX}"
        SHARED_LIB_SUFFIX="${IE_BUILD_POSTFIX}${CMAKE_SHARED_LIBRARY_SUFFIX}")

add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})
