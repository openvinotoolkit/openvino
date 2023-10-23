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

get_target_property(gtest_include_dirs gtest INTERFACE_INCLUDE_DIRECTORIES)
set_target_properties(gtest PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${gtest_include_dirs}")

get_target_property(gmock_include_dirs gtest INTERFACE_INCLUDE_DIRECTORIES)
set_target_properties(gmock PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
    "${gmock_include_dirs};${gmock_SOURCE_DIR}/include")

set(gtest_targets gtest gtest_main gmock gmock_main)

foreach(target IN LISTS gtest_targets)
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
    ov_disable_all_warnings(${target})
endforeach()

set_target_properties(${gtest_targets}
                      PROPERTIES FOLDER thirdparty)
