# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

#
# ov_check_pip_package(REQUIREMENT <single requirement>
#                      RESULT_VAR <result var name>
#                      [MESSAGE_MODE <WARNING | FATAL_ERROR>])
#
function(ov_check_pip_package)
    find_package(PythonInterp 3 REQUIRED)

    set(oneValueRequiredArgs
        REQUIREMENT             # Requirement-specifier to check
        RESULT_VAR              # Result varibale to set return code {ON | OFF}
        )
    set(oneValueOptionalArgs
        MESSAGE_MODE            # Set the type of message: { FATAL_ERROR | WARNING | ... }
        )
    set(multiValueArgs)

    cmake_parse_arguments(ARG "" "${oneValueRequiredArgs};${oneValueOptionalArgs}" "${multiValueArgs}" ${ARGN})

    foreach(argName ${oneValueRequiredArgs})
        if (NOT ARG_${argName})
            message(SEND_ERROR "Argument '${argName}' is required.")
        endif()
    endforeach()

    if(NOT ARG_MESSAGE_MODE)
        set(ARG_MESSAGE_MODE WARNING)
    endif()

    if(ARG_UNPARSED_ARGUMENTS)
        message(SEND_ERROR "Unexpected parameters have passed to the function: ${ARG_UNPARSED_ARGUMENTS}")
    endif()

    # quote '3.x' with \'3.x\'
    string(REPLACE "'" "\\'" REQ "${ARG_REQUIREMENT}")

    execute_process(
        COMMAND ${PYTHON_EXECUTABLE} -c "import pkg_resources ; pkg_resources.require('${REQ}')"
        RESULT_VARIABLE EXIT_CODE
        OUTPUT_VARIABLE OUTPUT_TEXT
        ERROR_VARIABLE ERROR_TEXT)

    if(NOT EXIT_CODE EQUAL 0)
        set(${ARG_RESULT_VAR} OFF PARENT_SCOPE)
        message(${ARG_MESSAGE_MODE} "Python requirement ${REQ} is missed, some functionality is disabled")
    else()
        set(${ARG_RESULT_VAR} ON PARENT_SCOPE)
    endif()
endfunction()

#
# ov_check_pip_packages(REQUIREMENTS_FILE <requirements.txt file>
#                       RESULT_VAR <result var name>
#                      [MESSAGE_MODE <WARNING | FATAL_ERROR>])
#
function(ov_check_pip_packages)
    find_package(PythonInterp 3 REQUIRED)

    set(oneValueOptionalArgs
        MESSAGE_MODE            # Set the type of message: { FATAL_ERROR | WARNING | ... }
        )
    set(oneValueRequiredArgs
        REQUIREMENTS_FILE       # File with requirement-specifiers to check
        RESULT_VAR              # Result varibale to set return code {TRUE | FALSE}
        )
    set(multiValueArgs)

    cmake_parse_arguments(ARG "" "${oneValueOptionalArgs};${oneValueRequiredArgs}" "${multiValueArgs}" ${ARGN})

    foreach(argName ${oneValueRequiredArgs})
        if (NOT ARG_${argName})
            message(SEND_ERROR "Argument '${argName}' is required.")
        endif()
    endforeach()

    if(NOT ARG_MESSAGE_MODE)
        set(ARG_MESSAGE_MODE WARNING)
    endif()

    if(ARG_UNPARSED_ARGUMENTS)
        message(SEND_ERROR "Unexpected parameters have passed to the function: ${ARG_UNPARSED_ARGUMENTS}")
    endif()

    execute_process(
        COMMAND ${PYTHON_EXECUTABLE} -c "import pkg_resources ; pkg_resources.require(open('${ARG_REQUIREMENTS_FILE}', mode='r'))"
        RESULT_VARIABLE EXIT_CODE
        OUTPUT_VARIABLE OUTPUT_TEXT
        ERROR_VARIABLE ERROR_TEXT)

    if(NOT EXIT_CODE EQUAL 0)
        set(${ARG_RESULT_VAR} OFF PARENT_SCOPE)
        message(${ARG_MESSAGE_MODE} "Python requirement file ${ARG_REQUIREMENTS_FILE} is not installed, some functionality is disabled")
    else()
        set(${ARG_RESULT_VAR} ON PARENT_SCOPE)
    endif()
endfunction()
