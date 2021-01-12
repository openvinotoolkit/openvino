# Copyright (C) 2018 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# [cmake:main]
cmake_minimum_required(VERSION 3.13)

project(InferenceEngineTemplatePlugin)

set(IE_MAIN_TEMPLATE_PLUGIN_SOURCE_DIR ${InferenceEngineTemplatePlugin_SOURCE_DIR})

find_package(InferenceEngineDeveloperPackage REQUIRED)

add_subdirectory(src)

if(ENABLE_TESTS)
    include(CTest)
    enable_testing()

    if(ENABLE_FUNCTIONAL_TESTS)
        add_subdirectory(tests/functional)
    endif()
endif()
# [cmake:main]

# install

# ATTENTION: uncomment to install component
# ie_cpack(template)
