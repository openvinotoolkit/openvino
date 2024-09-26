# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

#
# zeroApi
#

if(ENABLE_ZEROAPI_BACKEND)
    add_library(level-zero-ext INTERFACE)
    set_property(TARGET level-zero-ext APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/level-zero-ext/>)
    add_library(LevelZero::NPUExt ALIAS level-zero-ext)
    install(TARGETS level-zero-ext EXPORT "${PROJECT_NAME}Targets")
endif()
