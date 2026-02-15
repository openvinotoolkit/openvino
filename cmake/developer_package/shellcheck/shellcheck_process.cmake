# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

foreach(var SHELLCHECK_PROGRAM SHELL_SCRIPT SHELLCHECK_OUTPUT)
    if(NOT DEFINED ${var})
        message(FATAL_ERROR "${var} is not defined")
    endif()
endforeach()

set(rules "SC1091,SC2164,SC2162,SC1090")
execute_process(COMMAND ${SHELLCHECK_PROGRAM} --exclude=${rules} ${SHELL_SCRIPT}
                OUTPUT_VARIABLE error_message
                RESULT_VARIABLE exit_code
                OUTPUT_STRIP_TRAILING_WHITESPACE)

file(WRITE "${SHELLCHECK_OUTPUT}" "${error_message}")

if(NOT exit_code EQUAL 0)
    message(FATAL_ERROR "${error_message}")
endif()
