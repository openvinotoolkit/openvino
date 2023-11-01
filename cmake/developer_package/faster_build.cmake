# Copyright (C) 2018-2023 Intel Corporation
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

# deprecated

function(ie_faster_build)
    message(WARNING "ie_faster_build is deprecated, use ov_build_target_faster instead")
    ov_build_target_faster(${ARGV})
endfunction()
