# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

enable_testing()

if(NOT MSVC)
    ie_add_compiler_flags(-Wno-unused-variable)
endif()

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
