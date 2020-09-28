# Copyright (C) 2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

#####################################################################################################
#                                     SETUP GOOGLE TESTS                                            #
#####################################################################################################
set(gtest_force_shared_crt ON CACHE BOOL "disable static CRT for google test")
function(add_gtest_libraries)
    if (UNIX)
        ie_add_compiler_flags(-Wno-undef)
    endif ()

    add_subdirectory(gtest)

    get_target_property(gtest_include_dirs gtest INTERFACE_INCLUDE_DIRECTORIES)
    set_target_properties(gtest PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${gtest_include_dirs}")

    get_target_property(gmock_include_dirs gtest INTERFACE_INCLUDE_DIRECTORIES)
    set_target_properties(gmock PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${gmock_include_dirs};${gmock_SOURCE_DIR}/include")

    set_target_properties(gtest gtest_main gmock gmock_main
                          PROPERTIES FOLDER thirdparty)
endfunction()

add_gtest_libraries()

if (MSVC)
    set(PUGI pugixml_mt)
else ()
    set(PUGI pugixml)
endif ()

list(APPEND EXPORT_DEPENDENCIES
        ${PUGI}
        ${NGRAPH_LIBRARIES}
        gtest
        gtest_main
        )

set(TARGET_NAME commonTestUtils)

function(add_common_utils ADD_TARGET_NAME)
    # create target
    addIeTarget(
            NAME ${ADD_TARGET_NAME}
            TYPE STATIC
            ROOT ${CMAKE_CURRENT_SOURCE_DIR}
            EXCLUDED_SOURCE_DIRS
                ${CMAKE_CURRENT_SOURCE_DIR}/gtest
            ADD_CPPLINT
            DEVELOPER_PACKAGE
            EXPORT_DEPENDENCIES
                ${EXPORT_DEPENDENCIES}
    )

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
            ${IE_TESTS_ROOT}/ie_test_utils
            $<TARGET_PROPERTY:inference_engine,INTERFACE_INCLUDE_DIRECTORIES>
            $<TARGET_PROPERTY:pugixml,INTERFACE_INCLUDE_DIRECTORIES>
            PRIVATE
            $<TARGET_PROPERTY:inference_engine_plugin_api,INTERFACE_INCLUDE_DIRECTORIES>
            )

    target_compile_definitions(${ADD_TARGET_NAME} PUBLIC ${ARGN})

    target_link_libraries(${ADD_TARGET_NAME}
            PUBLIC
            ${EXPORT_DEPENDENCIES}
            )
endfunction()

add_common_utils(${TARGET_NAME})
add_common_utils(${TARGET_NAME}_s USE_STATIC_IE)
