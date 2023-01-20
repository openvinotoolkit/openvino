# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

enable_testing()

if(NOT MSVC)
    ie_add_compiler_flags(-Wno-missing-declarations)
    ie_add_compiler_flags(-Wno-sign-compare)
    ie_add_compiler_flags(-Wno-odr)
    ie_add_compiler_flags(-Wno-all)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    ie_add_compiler_flags(/wd4244)
    ie_add_compiler_flags(/wd4267)
    ie_add_compiler_flags(/wd4305)
    ie_add_compiler_flags(/wd4273)
    ie_add_compiler_flags(/wd4661)
    ie_add_compiler_flags(/wd4018)
    ie_add_compiler_flags(/wd4334)
endif()

add_subdirectory(deprecated/readers)
add_subdirectory(deprecated/helpers)

if(ENABLE_TESTS)
    add_subdirectory(unit)
    add_subdirectory(deprecated/unit)
endif()

if(ENABLE_FUNCTIONAL_TESTS)
    add_subdirectory(deprecated/functional)
    add_subdirectory(functional)
endif()
