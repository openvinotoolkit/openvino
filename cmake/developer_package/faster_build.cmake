# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include(CMakeParseArguments)

function(ov_build_target_faster TARGET_NAME)
    if(NOT ENABLE_FASTER_BUILD)
        return()
    endif()

    cmake_parse_arguments(FASTER_BUILD "UNITY" "" "PCH" ${ARGN})

    if(FASTER_BUILD_UNITY)
        set_target_properties(${TARGET_NAME} PROPERTIES UNITY_BUILD ON)
    endif()

    if(FASTER_BUILD_PCH)
        target_precompile_headers(${TARGET_NAME} ${FASTER_BUILD_PCH})
    endif()
endfunction()
