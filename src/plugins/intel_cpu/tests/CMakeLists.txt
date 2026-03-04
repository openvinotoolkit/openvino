# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # C4267, 4244 issues from oneDNN headers conversion from 'XXX' to 'YYY', possible loss of data
    ov_add_compiler_flags(/wd4018)
    ov_add_compiler_flags(/wd4267)
    ov_add_compiler_flags(/wd4244)
    # mkldnn headers: '<<': result of 32-bit shift implicitly converted to 64 bits
    ov_add_compiler_flags(/wd4334)
    # oneDNN arm64: unary minus operator applied to unsigned type, result still unsigned
    ov_add_compiler_flags(/wd4146)
endif()

add_subdirectory(unit)

if(ENABLE_FUNCTIONAL_TESTS)
    function(ov_cpu_func_tests)
        if(CMAKE_COMPILER_IS_GNUCXX)
            ov_add_compiler_flags(-Wno-unused-function)
            ov_add_compiler_flags(-Wno-parentheses)
            ov_add_compiler_flags(-Wno-unused-local-typedefs)
            ov_add_compiler_flags(-Wno-reorder)
            ov_add_compiler_flags(-Wno-comment)
            ov_add_compiler_flags(-Wno-unused-local-typedefs)
        endif()
        if(CMAKE_COMPILER_IS_GNUCXX OR OV_COMPILER_IS_CLANG AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
            ov_add_compiler_flags(-Wno-unused-but-set-variable)
        endif()

        add_subdirectory(functional)
    endfunction()

    ov_cpu_func_tests()
endif()
