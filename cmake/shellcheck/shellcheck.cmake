# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# TODO
set(IE_SHELLCHECK_REPORTS "${CMAKE_BINARY_DIR}/shellcheck")
# file(MAKE_DIRECTORY "${IE_SHELLCHECK_REPORTS}")

include(CMakeParseArguments)

find_host_program(shellcheck_PROGRAM NAMES shellcheck DOC "Path to shellcheck tool")

function(ie_shellcheck_process)
    if(NOT shellcheck_PROGRAM)
        message(WARNING "shellcheck tool is not found")
        return()
    endif()

    cmake_parse_arguments(IE_SHELLCHECK "" "DIRECTORY" "SKIP" ${ARGN})

    set(IE_SHELLCHECK_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/cmake/shellcheck/shellcheck_process.cmake")
    file(GLOB_RECURSE scripts "${IE_SHELLCHECK_DIRECTORY}/*.sh")
    foreach(script IN LISTS scripts)
        # check if we need to skip scripts
        unset(skip_script)
        foreach(skip_directory IN LISTS IE_SHELLCHECK_SKIP)
            if(script MATCHES "${skip_directory}/*")
                set(skip_script ON)
            endif()
        endforeach()
        if(skip_script)
            continue()
        endif()

        message("PROCESS ${script}")
        get_filename_component(script_name "${script}" NAME)
        set(output_file "${IE_SHELLCHECK_REPORTS}/${script_name}.txt")
        add_custom_command(OUTPUT ${output_file} 
                           COMMAND ${CMAKE_COMMAND}
                             -D IE_SHELLCHECK_PROGRAM=${shellcheck_PROGRAM}
                             -D IE_SHELL_SCRIPT=${script}
                             -D IE_SHELLCHECK_OUTPUT=${output_file}
                             -P ${IE_SHELLCHECK_SCRIPT}
                           DEPENDS ${script}
                           COMMENT "Check script ${script_name}"
                           VERBATIM)
        list(APPEND outputs ${output_file})
    endforeach()

    add_custom_target(ie_shellcheck DEPENDS ${outputs})
endfunction()
