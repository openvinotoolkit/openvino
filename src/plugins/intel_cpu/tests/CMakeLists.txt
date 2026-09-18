# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# Static oneDNN exports use the CPU-plugin-prefixed symbols. Apply the matching
# definitions to all CPU test targets without leaking them to third-party targets.
if(NOT BUILD_SHARED_LIBS)
    add_compile_definitions(${onednn_renamed_definitions})
endif()

# Suppress warnings in all test code (applies to unit and functional tests)
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    ov_add_compiler_flags(/wd4018)  # signed/unsigned mismatch
    ov_add_compiler_flags(/wd4267)  # conversion from 'size_t' to smaller type
    ov_add_compiler_flags(/wd4334)  # 'argument': cannot convert from 'type1' to 'type2'
    ov_add_compiler_flags(/wd4244)  # narrowing conversion
    ov_add_compiler_flags(/wd5051)  # attribute warning
elseif (OV_COMPILER_IS_INTEL_LLVM AND WIN32)
    ov_add_compiler_flags("/Wno-microsoft-include")
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
