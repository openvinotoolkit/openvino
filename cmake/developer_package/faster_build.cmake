# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include(CMakeParseArguments)

function(ov_build_target_faster TARGET_NAME)
    # ENABLE_FASTER_BUILD option enables usage of precompiled headers
    # ENABLE_UNITY_BUILD option enalbles unity build
    cmake_parse_arguments(FASTER_BUILD "UNITY" "PCH" "PCH_EXCLUDE" ${ARGN})

    if(FASTER_BUILD_UNITY AND ENABLE_UNITY_BUILD)
        set_target_properties(${TARGET_NAME} PROPERTIES UNITY_BUILD ON)
    endif()

    if(FASTER_BUILD_PCH AND ENABLE_FASTER_BUILD)
        target_precompile_headers(${TARGET_NAME} PRIVATE ${FASTER_BUILD_PCH})
        foreach(exclude_src IN LISTS FASTER_BUILD_PCH_EXCLUDE)
            set_source_files_properties(${exclude_src} PROPERTIES SKIP_PRECOMPILE_HEADERS ON)
        endforeach()
    endif()
endfunction()
