# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_subdirectory(ittapi EXCLUDE_FROM_ALL)
add_subdirectory(itt_collector EXCLUDE_FROM_ALL)
add_subdirectory(xbyak EXCLUDE_FROM_ALL)
add_subdirectory(zlib EXCLUDE_FROM_ALL)
add_subdirectory(cnpy EXCLUDE_FROM_ALL)

#
# Google Tests framework
#

function(add_gtest_libraries)
    set(gtest_force_shared_crt ON CACHE BOOL "disable static CRT for google test")
    if (UNIX)
        ie_add_compiler_flags(-Wno-undef)
        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            ie_add_compiler_flags(-Wno-deprecated-copy)
        endif()
    endif()

    set(BUILD_SHARED_LIBS OFF)
    set(INSTALL_GTEST OFF)
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

if(NGRAPH_UNIT_TEST_ENABLE OR ENABLE_TESTS)
    add_gtest_libraries()
endif()

#
# Export 3rdparty components
#

openvino_developer_export_targets(COMPONENT openvino_common TARGETS xbyak)
