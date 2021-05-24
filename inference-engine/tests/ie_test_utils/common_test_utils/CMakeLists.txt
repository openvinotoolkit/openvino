# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

#####################################################################################################
#                                     SETUP GOOGLE TESTS                                            #
#####################################################################################################
set(gtest_force_shared_crt ON CACHE BOOL "disable static CRT for google test")
function(add_gtest_libraries)
    if (UNIX)
        ie_add_compiler_flags(-Wno-undef)
        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            ie_add_compiler_flags(-Wno-deprecated-copy)
        endif()
    endif ()

    set(BUILD_SHARED_LIBS OFF)
    add_subdirectory(gtest EXCLUDE_FROM_ALL)

    get_target_property(gtest_include_dirs gtest INTERFACE_INCLUDE_DIRECTORIES)
    set_target_properties(gtest PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${gtest_include_dirs}")

    get_target_property(gmock_include_dirs gtest INTERFACE_INCLUDE_DIRECTORIES)
    set_target_properties(gmock PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${gmock_include_dirs};${gmock_SOURCE_DIR}/include")

    # If we have specified /Z7 option, remove -Zi option which comes from gtest
    if (WIN32)
        set(gtest_targets "gtest;gtest_main;gmock;gmock_main")
        foreach(target_name ${gtest_targets})
            if(TARGET "${target_name}")
                get_target_property(_target_cxx_flags ${target_name} COMPILE_FLAGS)
                if(_target_cxx_flags)
                    if(CMAKE_CXX_FLAGS_DEBUG MATCHES ".+/Z7.+" OR CMAKE_CXX_FLAGS_RELWITHDEBINFO MATCHES ".+/Z7.+")
                        string(REPLACE "-Zi" " " _target_cxx_flags ${_target_cxx_flags})
                        message(STATUS "Removing -Zi flag from target " ${target_name})
                        set_target_properties(${target_name} PROPERTIES COMPILE_FLAGS "${_target_cxx_flags}")
                    endif()
                endif()
            endif()
        endforeach()
    endif()
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
            EXCLUDED_SOURCE_PATHS
                ${CMAKE_CURRENT_SOURCE_DIR}/gtest
            ADD_CPPLINT
            DEVELOPER_PACKAGE
                inference_engine_tests
            EXPORT_DEPENDENCIES
                ${EXPORT_DEPENDENCIES}
            LINK_LIBRARIES
                PUBLIC
                    ${PUGI}
                    ${NGRAPH_LIBRARIES}
                    ${EXPORT_DEPENDENCIES}
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
            ${IE_TESTS_ROOT}/ie_test_utils
            $<TARGET_PROPERTY:inference_engine,INTERFACE_INCLUDE_DIRECTORIES>
            $<TARGET_PROPERTY:pugixml,INTERFACE_INCLUDE_DIRECTORIES>
        PRIVATE
            $<TARGET_PROPERTY:inference_engine_legacy,INTERFACE_INCLUDE_DIRECTORIES>
            $<TARGET_PROPERTY:inference_engine_plugin_api,INTERFACE_INCLUDE_DIRECTORIES>
            $<TARGET_PROPERTY:inference_engine_transformations,INTERFACE_INCLUDE_DIRECTORIES>
            )

    target_compile_definitions(${ADD_TARGET_NAME} PUBLIC ${ARGN})
endfunction()

add_common_utils(${TARGET_NAME})
add_common_utils(${TARGET_NAME}_s USE_STATIC_IE)
