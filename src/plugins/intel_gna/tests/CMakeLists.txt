# Copyright (C) 2022-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
enable_testing()

if(NOT MSVC)
    ie_add_compiler_flags(-Wno-missing-declarations)
    ie_add_compiler_flags(-Wno-sign-compare)
    ie_add_compiler_flags(-Wno-odr)
    ie_add_compiler_flags(-Wno-all)
endif()

if(ENABLE_TESTS)
    add_subdirectory(unit)
endif()

if(ENABLE_FUNCTIONAL_TESTS)
    add_subdirectory(functional)
endif()