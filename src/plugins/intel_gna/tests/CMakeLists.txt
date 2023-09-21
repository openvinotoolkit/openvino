# Copyright (C) 2022 Intel Corporation
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
    ov_add_compiler_flags(/wd4250)
else()
    ov_add_compiler_flags(-Wno-missing-declarations)
    ov_add_compiler_flags(-Wno-odr)
    ov_add_compiler_flags(-Wno-all)
endif()

if(ENABLE_TESTS)
    add_subdirectory(deprecated/readers)
    add_subdirectory(deprecated/helpers)

    add_subdirectory(unit)
    add_subdirectory(deprecated/unit)
endif()

if(ENABLE_FUNCTIONAL_TESTS)
    add_subdirectory(functional)
endif()
