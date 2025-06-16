# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include(CMakeParseArguments)

function(ov_build_target_faster TARGET_NAME)
    # ENABLE_FASTER_BUILD option enables usage of precompiled headers
    # ENABLE_UNITY_BUILD option enalbles unity build
    set(options PCH UNITY)
    set(oneValueArgs PCH_HEADER)
    set(multiValueArgs PCH_EXCLUDE)
    cmake_parse_arguments(PARSE_ARGV 0 FASTER_BUILD "${options}" "${oneValueArgs}" "${multiValueArgs}")

    if(FASTER_BUILD_UNITY AND ENABLE_UNITY_BUILD)
        set_target_properties(${TARGET_NAME} PROPERTIES UNITY_BUILD ON)
    endif()

    if(ENABLE_FASTER_BUILD)
        if (FASTER_BUILD_PCH_HEADER)
            target_precompile_headers(${TARGET_NAME} PRIVATE ${FASTER_BUILD_PCH_HEADER})
        elseif(FASTER_BUILD_PCH)
            get_target_property(pch_core_header_path openvino::util PCH_CORE_HEADER_PATH)
            if(NOT pch_core_header_path STREQUAL "pch_core_header_path-NOTFOUND")
                target_precompile_headers(${TARGET_NAME} PRIVATE ${pch_core_header_path})
            endif()
        endif()
        set_source_files_properties(${FASTER_BUILD_PCH_EXCLUDE} PROPERTIES SKIP_PRECOMPILE_HEADERS ON)
    endif()
endfunction()
