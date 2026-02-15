# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include(CMakeParseArguments)

find_host_program(shellcheck_PROGRAM NAMES shellcheck DOC "Path to shellcheck tool")

if(shellcheck_PROGRAM)
    execute_process(COMMAND "${shellcheck_PROGRAM}" --version
        RESULT_VARIABLE shellcheck_EXIT_CODE
        OUTPUT_VARIABLE shellcheck_VERSION_STRING)
    if(shellcheck_EXIT_CODE EQUAL 0)
        if(shellcheck_VERSION_STRING MATCHES "version: ([0-9]+)\.([0-9]+).([0-9]+)")
            set(shellcheck_VERSION "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}.${CMAKE_MATCH_3}" CACHE INTERNAL "shellcheck version")
        endif()
    endif()
endif()

function(ov_shellcheck_process)
    if(NOT shellcheck_PROGRAM)
        message(WARNING "shellcheck tool is not found")
        return()
    endif()

    cmake_parse_arguments(SHELLCHECK "" "DIRECTORY" "SKIP" ${ARGN})

    set(SHELLCHECK_SCRIPT "${OpenVINODeveloperScripts_DIR}/shellcheck/shellcheck_process.cmake")
    file(GLOB_RECURSE scripts "${SHELLCHECK_DIRECTORY}/*.sh")
    foreach(script IN LISTS scripts)
        # check if we need to skip scripts
        unset(skip_script)
        foreach(skip_directory IN LISTS SHELLCHECK_SKIP)
            if(script MATCHES "${skip_directory}/*")
                set(skip_script ON)
            endif()
        endforeach()
        if(skip_script)
            continue()
        endif()

        string(REPLACE "${SHELLCHECK_DIRECTORY}" "${CMAKE_BINARY_DIR}/shellcheck" output_file ${script})
        set(output_file "${output_file}.txt")
        get_filename_component(script_name "${script}" NAME)

        add_custom_command(OUTPUT ${output_file}
                           COMMAND ${CMAKE_COMMAND}
                             -D SHELLCHECK_PROGRAM=${shellcheck_PROGRAM}
                             -D SHELL_SCRIPT=${script}
                             -D SHELLCHECK_OUTPUT=${output_file}
                             -P ${SHELLCHECK_SCRIPT}
                           DEPENDS ${script} ${SHELLCHECK_SCRIPT}
                           COMMENT "Check script ${script_name}"
                           VERBATIM)
        list(APPEND outputs ${output_file})
    endforeach()

    add_custom_target(ov_shellcheck DEPENDS ${outputs})
endfunction()
