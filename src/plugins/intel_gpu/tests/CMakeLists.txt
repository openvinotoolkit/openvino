# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # Test code is not part of the released plugin DLL (not subject to BinSkim BA2007).
    # Keep warning suppression here to avoid mass test code changes.
    ov_add_compiler_flags(/wd4244)
    ov_add_compiler_flags(/wd4018)
endif()

add_subdirectory(common)
add_subdirectory(unit)
add_subdirectory(functional)
