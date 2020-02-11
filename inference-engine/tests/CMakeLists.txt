# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

#####################################################################################################
#                                     SETUP GOOGLE TESTS                                            #
#####################################################################################################

set (gtest_force_shared_crt ON CACHE BOOL "disable static CRT for google test")

function(add_gtest_libraries)
    if(UNIX)
        ie_add_compiler_flags(-Wno-undef)
    endif()

    add_subdirectory(libs/gtest)

    get_target_property(gtest_include_dirs gtest INTERFACE_INCLUDE_DIRECTORIES)
    set_target_properties(gtest PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${gtest_include_dirs}")

    get_target_property(gmock_include_dirs gtest INTERFACE_INCLUDE_DIRECTORIES)
    set_target_properties(gmock PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${gmock_include_dirs};${gmock_SOURCE_DIR}/include")
endfunction()

add_gtest_libraries()

#####################################################################################################
#                                         SETUP TESTS                                               #
#####################################################################################################

enable_testing()

# detecting regex support
if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.9)
    set(USE_BOOST_RE ON)
endif()

add_subdirectory(mock_engine)

add_subdirectory(helpers)

disable_deprecated_warnings()

if(ENABLE_TESTS)
  add_subdirectory(unit)
endif()
