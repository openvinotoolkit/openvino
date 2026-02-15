# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

find_package(RapidJSON QUIET)

if(NOT TARGET rapidjson)
    # sometimes RapidJSONConfig.cmake defines only RAPIDJSON_INCLUDE_DIRS
    add_library(rapidjson INTERFACE)

    if(RapidJSON_FOUND)
        if(TARGET RapidJSON)
            target_link_libraries(rapidjson INTERFACE RapidJSON)
        elseif(DEFINED RAPIDJSON_INCLUDE_DIRS)
            target_include_directories(rapidjson INTERFACE $<BUILD_INTERFACE:${RAPIDJSON_INCLUDE_DIRS}>)
        else()
            message(FATAL_ERROR "RapidJSON does not define RAPIDJSON_INCLUDE_DIRS nor RapidJSON / rapidjson targets")
        endif()
    else()
        target_include_directories(rapidjson INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/thirdparty>)
    endif()
endif()
