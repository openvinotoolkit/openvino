# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(NOT DEFINED IE_SHELLCHECK_PROGRAM)
    message(FATAL_ERROR "IE_SHELLCHECK_PROGRAM is not defined")
endif()

if(NOT DEFINED IE_SHELL_SCRIPT)
    message(FATAL_ERROR "IE_SHELL_SCRIPT is not defined")
endif()

if(NOT DEFINED IE_SHELLCHECK_OUTPUT)
    message(FATAL_ERROR "IE_SHELLCHECK_OUTPUT is not defined")
endif()

set(rules "SC1091,SC2164,SC2162,SC1090")
execute_process(COMMAND ${IE_SHELLCHECK_PROGRAM} --exclude=${rules} ${IE_SHELL_SCRIPT}
                OUTPUT_VARIABLE error_message
                RESULT_VARIABLE exit_code
                OUTPUT_STRIP_TRAILING_WHITESPACE)

file(WRITE "${IE_SHELLCHECK_OUTPUT}" "${error_message}")

if(NOT exit_code EQUAL 0)
    message(FATAL_ERROR "${error_message}")
endif()
