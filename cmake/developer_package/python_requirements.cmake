# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

#
# ov_check_pip_package(REQUIREMENT <single requirement>
#                      RESULT_VAR <result var name>
#                      [WARNING_MESSAGE <message>]
#                      [MESSAGE_MODE <WARNING | FATAL_ERROR | TRACE>])
#
function(ov_check_pip_package)
    find_host_package(Python3 QUIET COMPONENTS Interpreter)

    set(oneValueOptionalArgs
        MESSAGE_MODE            # Set the type of message: { FATAL_ERROR | WARNING | ... }
        WARNING_MESSAGE         # callback message
        )
    set(oneValueRequiredArgs
        REQUIREMENT             # Requirement-specifier to check
        RESULT_VAR              # Result varibale to set return code {ON | OFF}
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
    elseif(CMAKE_VERSION VERSION_LESS 3.15 AND ARG_MESSAGE_MODE STREQUAL "TRACE")
        set(ARG_MESSAGE_MODE WARNING)
    endif()

    if(ARG_UNPARSED_ARGUMENTS)
        message(SEND_ERROR "Unexpected parameters have passed to the function: ${ARG_UNPARSED_ARGUMENTS}")
    endif()

    # quote '3.x' with \'3.x\'
    string(REPLACE "'" "\\'" REQ "${ARG_REQUIREMENT}")

    if(Python3_Interpreter_FOUND)
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -c "import pkg_resources ; pkg_resources.require('${REQ}')"
            RESULT_VARIABLE EXIT_CODE
            OUTPUT_VARIABLE OUTPUT_TEXT
            ERROR_VARIABLE ERROR_TEXT)
    else()
        set(EXIT_CODE 1)
    endif()

    if(EXIT_CODE EQUAL 0)
        set(${ARG_RESULT_VAR} ON PARENT_SCOPE)
    else()
        set(${ARG_RESULT_VAR} OFF PARENT_SCOPE)
        message(${ARG_MESSAGE_MODE} "Python module '${REQ}' is missed, ${ARG_WARNING_MESSAGE}")
    endif()
endfunction()

#
# ov_check_pip_packages(REQUIREMENTS_FILE <requirements.txt file>
#                       RESULT_VAR <result var name>
#                      [WARNING_MESSAGE <message>]
#                      [MESSAGE_MODE <WARNING | FATAL_ERROR | TRACE>])
#
function(ov_check_pip_packages)
    find_host_package(Python3 QUIET COMPONENTS Interpreter)

    set(oneValueOptionalArgs
        MESSAGE_MODE            # Set the type of message: { FATAL_ERROR | WARNING | ... }
        WARNING_MESSAGE         # callback message
        )
    set(oneValueRequiredArgs
        REQUIREMENTS_FILE       # File with requirement-specifiers to check
        RESULT_VAR              # Result varibale to set return code {ON | OFF}
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
    elseif(CMAKE_VERSION VERSION_LESS 3.15 AND ARG_MESSAGE_MODE STREQUAL "TRACE")
        set(ARG_MESSAGE_MODE WARNING)
    endif()

    if(ARG_UNPARSED_ARGUMENTS)
        message(SEND_ERROR "Unexpected parameters have passed to the function: ${ARG_UNPARSED_ARGUMENTS}")
    endif()

    if(Python3_Interpreter_FOUND)
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -c "
from check_python_requirements import check_python_requirements ;
check_python_requirements('${ARG_REQUIREMENTS_FILE}') ;
            "
            WORKING_DIRECTORY "${OpenVINODeveloperScripts_DIR}"
            RESULT_VARIABLE EXIT_CODE
            OUTPUT_VARIABLE OUTPUT_TEXT
            ERROR_VARIABLE ERROR_TEXT)
    else()
        set(EXIT_CODE 1)
    endif()

    if(EXIT_CODE EQUAL 0)
        set(${ARG_RESULT_VAR} ON PARENT_SCOPE)
    else()
        set(${ARG_RESULT_VAR} OFF PARENT_SCOPE)
        message(${ARG_MESSAGE_MODE} "Python requirement file ${ARG_REQUIREMENTS_FILE} is not installed, ${ARG_WARNING_MESSAGE}")
    endif()
endfunction()
