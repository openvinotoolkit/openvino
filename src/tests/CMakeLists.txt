# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(OV_TESTS_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

enable_testing()

ov_option(ENABLE_CONFORMANCE_PGQL "Enables support of PostgreSQL-based reporting from test tools" OFF)
mark_as_advanced(FORCE ENABLE_CONFORMANCE_PGQL)

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    ov_add_compiler_flags(/wd4244)
    ov_add_compiler_flags(/wd4267)
    ov_add_compiler_flags(/wd4305)
    ov_add_compiler_flags(/wd4018)
    ov_add_compiler_flags(/wd4050)
    ov_add_compiler_flags(/wd4250)
    ov_add_compiler_flags(/wd4334)
    ov_add_compiler_flags(/wd4661)
    ov_add_compiler_flags(/wd4273)
    ov_add_compiler_flags(/wd4309)
    ov_add_compiler_flags(/wd4804)
else()
    ov_add_compiler_flags(-Wno-missing-declarations)
    ov_add_compiler_flags(-Wno-odr)
    ov_add_compiler_flags(-Wno-all)
endif()

ov_deprecated_no_errors()

add_subdirectory(ov_helpers)
add_subdirectory(test_utils)

if(ENABLE_FUNCTIONAL_TESTS)
    add_subdirectory(functional)
endif()
