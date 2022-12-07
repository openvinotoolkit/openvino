# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

find_program(rpmlint_PROGRAM NAMES rpmlint DOC "Path to rpmlint tool")
if(NOT rpmlint_PROGRAM)
    message(WARNING "Failed to find 'rpmlint' tool, use 'sudo dnf / yum install rpmlint' to install it")
    return()
endif()

set(rpmlint_passed ON)

foreach(rpm_file IN LISTS CPACK_PACKAGE_FILES)
    get_filename_component(rpm_name "${rpm_file}" NAME)
    get_filename_component(dir_name "${rpm_file}" DIRECTORY)
    get_filename_component(dir_name "${dir_name}/../../../../rpmlint" ABSOLUTE)

    set(rpmlint_overrides "${dir_name}/${rpm_name}.rpmlintrc")
    if(EXISTS "${rpmlint_overrides}")
        set(file_option --file "${rpmlint_overrides}")
    endif()

    execute_process(COMMAND "${rpmlint_PROGRAM}" --strict ${file_option} ${rpm_file}
                    RESULT_VARIABLE rpmlint_exit_code
                    OUTPUT_VARIABLE rpmlint_output)

    if(NOT rpmlint_exit_code EQUAL 0)
        message("Package ${rpm_name}:")
        message("${rpmlint_output}")
        set(rpmlint_passed OFF)
    endif()

    unset(file_option)
endforeach()

if(NOT rpmlint_passed)
    message(FATAL_ERROR "rpmlint has found some mistakes. You can get more info regarding issues on site https://fedoraproject.org/wiki/Common_Rpmlint_issues")
endif()
