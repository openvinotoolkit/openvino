# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# [cmake:main]
cmake_minimum_required(VERSION 3.13)

project(OpenVINOTemplatePlugin)

find_package(OpenVINODeveloperPackage REQUIRED)

ov_option(ENABLE_TEMPLATE_REGISTRATION "Enables registration of TEMPLATE plugin" OFF)

if(CMAKE_COMPILER_IS_GNUCXX)
    ov_add_compiler_flags(-Wall)
endif()

add_subdirectory(src)

if(ENABLE_TESTS)
    include(CTest)
    enable_testing()

    # Check Nix-specific override for template functional tests
    if(DEFINED ENABLE_TEMPLATE_FUNCTIONAL_TESTS_OVERRIDE)
        set(TEMPLATE_FUNC_TESTS_ENABLED ${ENABLE_TEMPLATE_FUNCTIONAL_TESTS_OVERRIDE})
    else()
        set(TEMPLATE_FUNC_TESTS_ENABLED ${ENABLE_FUNCTIONAL_TESTS})
    endif()

    if(TEMPLATE_FUNC_TESTS_ENABLED)
        add_subdirectory(tests/functional)
    else()
        message(STATUS "Template functional tests disabled (Nix override or ENABLE_FUNCTIONAL_TESTS=OFF)")
    endif()
endif()

# install

if(OpenVINODeveloperPackage_FOUND)
    ov_cpack(template)
endif()
# [cmake:main]
