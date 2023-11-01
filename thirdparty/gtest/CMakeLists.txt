# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

#
# Google Tests framework
#

set(CMAKE_DISABLE_FIND_PACKAGE_PythonInterp ON)
set(gtest_force_shared_crt ON CACHE BOOL "disable static CRT for google test")

set(BUILD_SHARED_LIBS OFF)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)

add_subdirectory(gtest EXCLUDE_FROM_ALL)

function(_ov_gtest_filter_install_interface TARGET TYPE)
    set(final_include_dirs "$<INSTALL_INTERFACE:developer_package/include/${TYPE}>")

    get_target_property(include_dirs ${TARGET} INTERFACE_INCLUDE_DIRECTORIES)
    foreach(include_dir IN LISTS include_dirs)
        if(NOT include_dir MATCHES ".*INSTALL_INTERFACE.*")
            # remove leading and trailing parts of generator expressions
            string(REPLACE "$<BUILD_INTERFACE:" "" include_dir "${include_dir}")
            string(REPLACE ">" "" include_dir "${include_dir}")
            # wrap to BUILD_INTERFACE again
            list(APPEND final_include_dirs "$<BUILD_INTERFACE:${include_dir}>")
        endif()
    endforeach()

    set_target_properties(${TARGET} PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${final_include_dirs}"
        INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${final_include_dirs}")
endfunction()

_ov_gtest_filter_install_interface(gtest gtest)
_ov_gtest_filter_install_interface(gtest_main gtest)
_ov_gtest_filter_install_interface(gmock gmock)
_ov_gtest_filter_install_interface(gmock_main gmock)

foreach(target gtest gtest_main gmock gmock_main)
    # If we have specified /Z7 option, remove -Zi option which comes from gtest
    if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        get_target_property(_target_cxx_flags ${target} COMPILE_OPTIONS)
        if(_target_cxx_flags)
            if(CMAKE_CXX_FLAGS_DEBUG MATCHES ".+/Z7.+" OR CMAKE_CXX_FLAGS_RELWITHDEBINFO MATCHES ".+/Z7.+")
                string(REPLACE "-Zi" " " _target_cxx_flags ${_target_cxx_flags})
                message(STATUS "Removing -Zi flag from target " ${target})
                set_target_properties(${target} PROPERTIES COMPILE_OPTIONS "${_target_cxx_flags}")
            endif()
        endif()
    elseif(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "^(Apple)?Clang$" OR
        CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
        target_compile_options(${target} PRIVATE -Wno-undef)
        if(CMAKE_COMPILER_IS_GNUCXX)
            target_compile_options(${target} PRIVATE -Wno-deprecated-copy)
        endif()
    endif()
    # disable warnings
    ov_disable_all_warnings(${target})
    set_target_properties(${target} PROPERTIES FOLDER thirdparty)
endforeach()
