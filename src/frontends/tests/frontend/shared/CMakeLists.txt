# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "frontend_shared_test_classes")

set(LIBRARY_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/basic_api.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/conversion.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/conversion_with_reference.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/convert_model.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cut_specific_model.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lib_close.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/library_extension.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/load_from.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op_extension.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op_fuzzy.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/partial_shape.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/set_element_type.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/telemetry.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils.cpp
)
set(LIBRARY_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basic_api.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/conversion_extension.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/conversion_with_reference.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/convert_model.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/cut_specific_model.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/lib_close.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/library_extension.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/load_from.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/op_extension.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/op_fuzzy.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/partial_shape.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/set_element_type.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/telemetry.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/utils.hpp
)

add_library(${TARGET_NAME} STATIC EXCLUDE_FROM_ALL ${LIBRARY_SRC} ${LIBRARY_HEADERS})

add_subdirectory(gtest_main_manifest)
add_subdirectory(test_builtin_extensions)

target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../..)

target_link_libraries(${TARGET_NAME}
        PUBLIC
                openvino::offline_transformations
                common_test_utils
                func_test_utils
                openvino::util
                openvino::runtime
        PRIVATE
                openvino::cnpy)

add_dependencies(${TARGET_NAME} test_builtin_extensions)

target_compile_definitions(${TARGET_NAME}
    PRIVATE
        SHARED_LIB_PREFIX="${CMAKE_SHARED_LIBRARY_PREFIX}"
        SHARED_LIB_SUFFIX="${OV_BUILD_POSTFIX}${CMAKE_SHARED_LIBRARY_SUFFIX}")

ov_add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})

ov_build_target_faster(${TARGET_NAME} PCH)
