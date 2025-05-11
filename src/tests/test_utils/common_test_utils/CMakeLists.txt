# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

function(add_common_utils ADD_TARGET_NAME)
    list(APPEND TARGET_EXCLUDED_SOURCE_PATHS
        ${CMAKE_CURRENT_SOURCE_DIR}/tests
    )
    if(NOT ENABLE_CONFORMANCE_PGQL)
        list(APPEND TARGET_EXCLUDED_SOURCE_PATHS
            ${CMAKE_CURRENT_SOURCE_DIR}/include/common_test_utils/postgres_link.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/include/common_test_utils/postgres_helpers.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/src/postgres_link.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/src/postgres_helpers.cpp
        )
    endif()

    # create target
    ov_add_target(
            NAME ${ADD_TARGET_NAME}
            TYPE STATIC
            ROOT ${CMAKE_CURRENT_SOURCE_DIR}
            EXCLUDED_SOURCE_PATHS
                ${TARGET_EXCLUDED_SOURCE_PATHS}
            ADD_CLANG_FORMAT
            DEPENDENCIES
                openvino_template_plugin
            LINK_LIBRARIES
                PUBLIC
                    gtest
                    gtest_main
                    gmock
                    openvino::runtime
                    openvino::runtime::dev
                PRIVATE
                    openvino::util
                    openvino::shape_inference
                INCLUDES
                    PUBLIC
                        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
                    PRIVATE
                        "${CMAKE_CURRENT_SOURCE_DIR}/src"
                        "${OpenVINO_SOURCE_DIR}/src/plugins/template/include"
    )

    ov_developer_package_export_targets(TARGET ${ADD_TARGET_NAME}
                                        INSTALL_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/include/")

    if(ENABLE_CONFORMANCE_PGQL)
        target_compile_definitions(${ADD_TARGET_NAME} PUBLIC ENABLE_CONFORMANCE_PGQL)
    endif()

    # USE_STATIC_IE is passed
    if(ARGN)
        target_link_libraries(${ADD_TARGET_NAME} PRIVATE openvino_runtime_s)
    endif()

    ov_build_target_faster(${ADD_TARGET_NAME}
        UNITY
        PCH PRIVATE "src/precomp.hpp"
    )

    target_include_directories(${ADD_TARGET_NAME}
        PUBLIC
            $<TARGET_PROPERTY:openvino::runtime,INTERFACE_INCLUDE_DIRECTORIES>
        PRIVATE
            $<TARGET_PROPERTY:openvino::runtime::dev,INTERFACE_INCLUDE_DIRECTORIES>)
    target_include_directories(${ADD_TARGET_NAME} SYSTEM PUBLIC "$<BUILD_INTERFACE:${OV_TESTS_ROOT}/test_utils>")

    target_compile_definitions(${ADD_TARGET_NAME} PUBLIC ${ARGN})
endfunction()

# Keep old name so that library can be used from NPU repo
set(TARGET_NAME commonTestUtils)
set(NEW_TARGET_NAME common_test_utils)

add_common_utils(${TARGET_NAME})
add_common_utils(${TARGET_NAME}_s USE_STATIC_IE)

# Add new names to use them from openvino repo
add_library(${NEW_TARGET_NAME} ALIAS ${TARGET_NAME})
add_library(${NEW_TARGET_NAME}_s ALIAS ${TARGET_NAME}_s)

add_subdirectory(tests)
