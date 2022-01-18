# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

enable_testing()

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
