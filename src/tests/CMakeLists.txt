# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(IE_TESTS_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

enable_testing()

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    ie_add_compiler_flags(/wd4244)
    ie_add_compiler_flags(/wd4267)
    ie_add_compiler_flags(/wd4305)
    ie_add_compiler_flags(/wd4018)
    ie_add_compiler_flags(/wd4050)
    ie_add_compiler_flags(/wd4250)
    ie_add_compiler_flags(/wd4334)
    ie_add_compiler_flags(/wd4661)
    ie_add_compiler_flags(/wd4273)
    ie_add_compiler_flags(/wd4309)
    ie_add_compiler_flags(/wd4804)
else()
    ie_add_compiler_flags(-Wno-missing-declarations)
    ie_add_compiler_flags(-Wno-odr)
    ie_add_compiler_flags(-Wno-all)
endif()

add_subdirectory(ngraph_helpers)
add_subdirectory(ie_test_utils)
add_subdirectory(util)
add_subdirectory(engines_util)

if(ENABLE_FUNCTIONAL_TESTS)
    add_subdirectory(functional)
endif()
