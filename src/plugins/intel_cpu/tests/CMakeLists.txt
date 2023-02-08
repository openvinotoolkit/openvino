# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_subdirectory(unit)

if(ENABLE_FUNCTIONAL_TESTS)
    function(ov_cpu_func_tests)
        if(CMAKE_COMPILER_IS_GNUCXX)
            ie_add_compiler_flags(-Wno-sign-compare)
            ie_add_compiler_flags(-Wno-unused-function)
            ie_add_compiler_flags(-Wno-parentheses)
            ie_add_compiler_flags(-Wno-unused-local-typedefs)
            ie_add_compiler_flags(-Wno-reorder)
            ie_add_compiler_flags(-Wno-comment)
            ie_add_compiler_flags(-Wno-unused-but-set-variable)
            ie_add_compiler_flags(-Wno-unused-local-typedefs)
        endif()

        add_subdirectory(functional)
    endfunction()

    ov_cpu_func_tests()
endif()
