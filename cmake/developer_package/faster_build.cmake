# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include(CMakeParseArguments)

function(ie_faster_build TARGET_NAME)
    if(NOT ENABLE_FASTER_BUILD)
        return()
    endif()

    cmake_parse_arguments(IE_FASTER_BUILD "UNITY" "" "PCH" ${ARGN})

    if(IE_FASTER_BUILD_UNITY)
        set_target_properties(${TARGET_NAME}
            PROPERTIES
                UNITY_BUILD ON
        )
    endif()

    if(IE_FASTER_BUILD_PCH)
        target_precompile_headers(${TARGET_NAME}
            ${IE_FASTER_BUILD_PCH}
        )
    endif()
endfunction()
