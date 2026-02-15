# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

find_program(rpmlint_PROGRAM NAMES rpmlint DOC "Path to rpmlint tool")
if(NOT rpmlint_PROGRAM)
    message(WARNING "Failed to find 'rpmlint' tool, use 'sudo dnf / yum install rpmlint' to install it")
    return()
endif()

set(rpmlint_passed ON)

execute_process(COMMAND "${rpmlint_PROGRAM}" --version
                RESULT_VARIABLE rpmlint_exit_code
                OUTPUT_VARIABLE rpmlint_version)

if(NOT rpmlint_exit_code EQUAL 0)
    message(FATAL_ERROR "Failed to get ${rpmlint_PROGRAM} version. Output is '${rpmlint_version}'")
endif()

if(rpmlint_version MATCHES "([0-9]+)\.([0-9]+)")
    set(rpmlint_version "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}")
else()
    message(FATAL_ERROR "Failed to parse rpmlint version '${rpmlint_version}'")
endif()

if(rpmlint_version VERSION_GREATER_EQUAL 2.0)
    set(rpmlint_has_strict_option ON)
endif()

foreach(rpm_file IN LISTS CPACK_PACKAGE_FILES)
    get_filename_component(rpm_name "${rpm_file}" NAME)
    get_filename_component(dir_name "${rpm_file}" DIRECTORY)
    get_filename_component(dir_name "${dir_name}/../../../../rpmlint" ABSOLUTE)

    set(rpmlint_overrides "${dir_name}/${rpm_name}.rpmlintrc")
    if(EXISTS "${rpmlint_overrides}")
        set(rpmlint_options --file "${rpmlint_overrides}")
    endif()
    if(rpmlint_has_strict_option)
        list(APPEND rpmlint_options --strict)
    endif()

    execute_process(COMMAND "${rpmlint_PROGRAM}" ${rpmlint_options} ${rpm_file}
                    RESULT_VARIABLE rpmlint_exit_code
                    OUTPUT_VARIABLE rpmlint_output)

    if(NOT rpmlint_exit_code EQUAL 0 OR NOT rpmlint_has_strict_option)
        message("Package ${rpm_name}:")
        message("${rpmlint_output}")
        if(rpmlint_has_strict_option)
            set(rpmlint_passed OFF)
        endif()
    endif()

    unset(rpmlint_options)
endforeach()

if(NOT rpmlint_passed)
    message(FATAL_ERROR "rpmlint has found some mistakes. You can get more info regarding issues on site https://fedoraproject.org/wiki/Common_Rpmlint_issues")
endif()
