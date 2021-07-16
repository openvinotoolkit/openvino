# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

function(add_common_utils ADD_TARGET_NAME)
    # create target
    addIeTarget(
            NAME ${ADD_TARGET_NAME}
            TYPE STATIC
            ROOT ${CMAKE_CURRENT_SOURCE_DIR}
            EXCLUDED_SOURCE_PATHS
                ${CMAKE_CURRENT_SOURCE_DIR}/gtest
            ADD_CPPLINT
            DEVELOPER_PACKAGE
                inference_engine_tests
            LINK_LIBRARIES
                PUBLIC
                    ngraph
                    gtest
                    gtest_main
                    inference_engine_transformations
    )

    # USE_STATIC_IE is passed
    if(ARGN)
        target_link_libraries(${ADD_TARGET_NAME} PRIVATE inference_engine_s)
    else()
        target_link_libraries(${ADD_TARGET_NAME} PRIVATE inference_engine_legacy)
    endif()

    ie_faster_build(${ADD_TARGET_NAME}
        UNITY
        PCH PRIVATE "precomp.hpp"
    )

    # detecting regex support
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.9)
        set(USE_BOOST_RE ON)
    endif()
    if (USE_BOOST_RE)
        target_compile_definitions(${ADD_TARGET_NAME} PUBLIC USE_BOOST_RE)

        debug_message(STATUS "Adding boost dependency")
        if (VERBOSE_BUILD)
            set(Boost_DEBUG on)
        endif ()
        find_package(Boost REQUIRED COMPONENTS regex)
        target_link_libraries(${ADD_TARGET_NAME} PUBLIC ${Boost_REGEX_LIBRARY})
        target_include_directories(${ADD_TARGET_NAME} PUBLIC ${Boost_INCLUDE_DIRS})
    endif ()

    target_include_directories(${ADD_TARGET_NAME}
        PUBLIC
            $<TARGET_PROPERTY:inference_engine,INTERFACE_INCLUDE_DIRECTORIES>
        PRIVATE
            $<TARGET_PROPERTY:inference_engine_legacy,INTERFACE_INCLUDE_DIRECTORIES>
            $<TARGET_PROPERTY:inference_engine_plugin_api,INTERFACE_INCLUDE_DIRECTORIES>
            $<TARGET_PROPERTY:inference_engine_transformations,INTERFACE_INCLUDE_DIRECTORIES>
            )
    target_include_directories(${ADD_TARGET_NAME} SYSTEM PUBLIC ${IE_TESTS_ROOT}/ie_test_utils)

    target_compile_definitions(${ADD_TARGET_NAME} PUBLIC ${ARGN})
endfunction()

set(TARGET_NAME commonTestUtils)

add_common_utils(${TARGET_NAME})
add_common_utils(${TARGET_NAME}_s USE_STATIC_IE)
