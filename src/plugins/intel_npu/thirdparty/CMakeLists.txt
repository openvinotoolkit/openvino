# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

#
# zeroApi
#

if(ENABLE_NPU_PLUGIN_ENGINE)
    add_library(level-zero-ext INTERFACE)
    set_property(TARGET level-zero-ext APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/level-zero-ext/>)
    add_library(LevelZero::NPUExt ALIAS level-zero-ext)
endif()

#
# yaml-cpp
#

if(ENABLE_INTEL_NPU_PROTOPIPE)
    set(YAML_BUILD_SHARED_LIBS OFF)
    add_subdirectory(yaml-cpp EXCLUDE_FROM_ALL)
    # NB: need to disable all warnings because some
    # OpenVINO CI builds may treat warnings as errors
    ov_disable_all_warnings(yaml-cpp)
endif()
