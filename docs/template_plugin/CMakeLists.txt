# Copyright (C) 2018 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# [cmake:main]
if (APPLE)
    # due to https://cmake.org/cmake/help/v3.12/policy/CMP0068.html
    cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
else()
    cmake_minimum_required(VERSION 3.7.2 FATAL_ERROR)
endif()

project(InferenceEngineTemplatePlugin)

set(IE_MAIN_TEMPLATE_PLUGIN_SOURCE_DIR ${InferenceEngineTemplatePlugin_SOURCE_DIR})

find_package(InferenceEngineDeveloperPackage REQUIRED)

add_subdirectory(src)

if(ENABLE_TESTS)
	include(CTest)
	enable_testing()

	if(ENABLE_FUNCTIONAL_TESTS)
	    add_subdirectory(tests_deprecated/functional)
	    add_subdirectory(tests/functional)
	endif()

	if(ENABLE_BEH_TESTS)
	    add_subdirectory(tests_deprecated/behavior)
	endif()
endif()
# [cmake:main]

# install

# ATTENTION: uncomment to install component
# ie_cpack(template)
