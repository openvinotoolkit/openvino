# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

function(add_common_utils ADD_TARGET_NAME)
    # create target
    addIeTarget(
            NAME ${ADD_TARGET_NAME}
            TYPE STATIC
            ROOT ${CMAKE_CURRENT_SOURCE_DIR}
            ADD_CPPLINT
            DEVELOPER_PACKAGE
                tests
            LINK_LIBRARIES
                PUBLIC
                    gtest
                    gtest_main
                    gmock
                    ngraphFunctions
                    openvino::runtime
                    openvino::runtime::dev
                    openvino::util
                PRIVATE
                    openvino::util
                    ov_shape_inference
    )

    # USE_STATIC_IE is passed
    if(ARGN)
        target_link_libraries(${ADD_TARGET_NAME} PRIVATE inference_engine_s)
    endif()

    ie_faster_build(${ADD_TARGET_NAME}
        UNITY
        PCH PRIVATE "include/common_test_utils/precomp.hpp"
    )

    # detecting regex support
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.9)
        set(USE_BOOST_RE ON)
    endif()
    if (USE_BOOST_RE)
        target_compile_definitions(${ADD_TARGET_NAME} PUBLIC USE_BOOST_RE)

        debug_message(STATUS "Adding boost dependency")
        if (CMAKE_VERBOSE_MAKEFILE)
            set(Boost_DEBUG on)
        endif ()
        find_package(Boost REQUIRED COMPONENTS regex)
        target_link_libraries(${ADD_TARGET_NAME} PUBLIC ${Boost_REGEX_LIBRARY})
        target_include_directories(${ADD_TARGET_NAME} PUBLIC ${Boost_INCLUDE_DIRS})
    endif ()

    target_include_directories(${ADD_TARGET_NAME}
        PUBLIC
            $<TARGET_PROPERTY:openvino::runtime,INTERFACE_INCLUDE_DIRECTORIES>
        PRIVATE
            $<TARGET_PROPERTY:openvino::runtime::dev,INTERFACE_INCLUDE_DIRECTORIES>)
    target_include_directories(${ADD_TARGET_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

    target_compile_definitions(${ADD_TARGET_NAME} PUBLIC ${ARGN})
endfunction()

set(TARGET_NAME commonTestUtils)

add_common_utils(${TARGET_NAME})
add_common_utils(${TARGET_NAME}_s USE_STATIC_IE)

openvino_developer_export_targets(COMPONENT tests TARGETS ${TARGET_NAME})
