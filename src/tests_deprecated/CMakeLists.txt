# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

enable_testing()


if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    ie_add_compiler_flags(/wd4244)
    ie_add_compiler_flags(/wd4267)
    ie_add_compiler_flags(/wd4305)
    ie_add_compiler_flags(/wd4273)
    ie_add_compiler_flags(/wd4661)
    ie_add_compiler_flags(/wd4018)
    ie_add_compiler_flags(/wd4334)
endif()

if(NOT MSVC)
    ie_add_compiler_flags(-Wno-missing-declarations)
    ie_add_compiler_flags(-Wno-sign-compare)
    ie_add_compiler_flags(-Wno-all)
    ie_add_compiler_flags(-Wno-unused-variable)
    ie_add_compiler_flags(-Wno-unused-function)
endif()

add_subdirectory(readers)
add_subdirectory(helpers)

if (ENABLE_GAPI_TESTS)
    add_subdirectory(fluid_preproc)
endif()

if (ENABLE_FUNCTIONAL_TESTS)
    add_subdirectory(functional)
endif()

if (ENABLE_BEH_TESTS)
    add_subdirectory(behavior)
endif()

if(ENABLE_TESTS)
    add_subdirectory(unit)
endif()
