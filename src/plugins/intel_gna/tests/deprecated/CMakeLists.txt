# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

enable_testing()

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    ov_add_compiler_flags(/wd4244)
    ov_add_compiler_flags(/wd4267)
    ov_add_compiler_flags(/wd4305)
    ov_add_compiler_flags(/wd4273)
    ov_add_compiler_flags(/wd4661)
    ov_add_compiler_flags(/wd4018)
    ov_add_compiler_flags(/wd4334)
endif()

if(NOT MSVC)
    ov_add_compiler_flags(-Wno-missing-declarations)
    ov_add_compiler_flags(-Wno-all)
    ov_add_compiler_flags(-Wno-unused-variable)
    ov_add_compiler_flags(-Wno-unused-function)
endif()

add_subdirectory(readers)
add_subdirectory(helpers)

if (ENABLE_FUNCTIONAL_TESTS)
    add_subdirectory(functional)
endif()

if(ENABLE_TESTS)
    add_subdirectory(unit)
endif()
