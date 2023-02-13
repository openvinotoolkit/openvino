# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

#
# ov_check_pip_package(REQUIREMENT <single requirement>
#                      RESULT_VAR <result var name>
#                      [WARNING_MESSAGE <message>]
#                      [MESSAGE_MODE <WARNING | FATAL_ERROR | TRACE>])
#
function(ov_check_pip_package)
    find_host_package(PythonInterp 3 QUIET)

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

    if(PYTHONINTERP_FOUND)
        execute_process(
            COMMAND ${PYTHON_EXECUTABLE} -c "import pkg_resources ; pkg_resources.require('${REQ}')"
            RESULT_VARIABLE EXIT_CODE
            OUTPUT_VARIABLE OUTPUT_TEXT
            ERROR_VARIABLE ERROR_TEXT)
    endif()

    if(NOT EXIT_CODE EQUAL 0)
        set(${ARG_RESULT_VAR} OFF PARENT_SCOPE)
        message(${ARG_MESSAGE_MODE} "Python module '${REQ}' is missed, ${ARG_WARNING_MESSAGE}")
    else()
        set(${ARG_RESULT_VAR} ON PARENT_SCOPE)
    endif()
endfunction()

#
# ov_check_pip_packages(REQUIREMENTS_FILE <requirements.txt file>
#                       RESULT_VAR <result var name>
#                      [WARNING_MESSAGE <message>]
#                      [MESSAGE_MODE <WARNING | FATAL_ERROR | TRACE>])
#
function(ov_check_pip_packages)
    find_host_package(PythonInterp 3 QUIET)

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

    if(PYTHONINTERP_FOUND)
        execute_process(
            COMMAND ${PYTHON_EXECUTABLE} -c "
import pkg_resources ;
import re ;
import os ;
constraints = {} ;
constraints_path = None ;
requirements = [] ;
with open('${ARG_REQUIREMENTS_FILE}') as f:
    raw_requirements = f.readlines() ;
for line in raw_requirements:
    if line.startswith('-c'):
        constraints_path = os.path.join(os.path.dirname('${ARG_REQUIREMENTS_FILE}'), line.split(' ')[1][:-1]) ;
if constraints_path: 
    with open(constraints_path) as f:
        raw_constraints = f.readlines() ;
    for line in raw_constraints:
        if line.startswith('#') or line=='\\n':
            continue ;
        line = line.replace('\\n', '') ;
        package, delimiter, constraint = re.split('(~|=|<|>|;)', line, maxsplit=1) ;
        if constraints.get(package) is None:
            constraints[package] = [delimiter + constraint] ;
        else:
            constraints[package].extend([delimiter + constraint]) ;
    for line in raw_requirements:
        if line.startswith(('#', '-c')):
            continue ;
        line = line.replace('\\n', '') ;
        if re.search('\\W', line):
            requirements.append(line) ;
        else:
            constraint = constraints.get(line) ;
            if constraint:
                for marker in constraint: 
                    requirements.append(line+marker) ;
            else:
                requirements.append(line) ;
else:
    requirements = raw_requirements ;
pkg_resources.require(requirements)"
            RESULT_VARIABLE EXIT_CODE
            OUTPUT_VARIABLE OUTPUT_TEXT
            ERROR_VARIABLE ERROR_TEXT)
    endif()

    if(NOT EXIT_CODE EQUAL 0)
        set(${ARG_RESULT_VAR} OFF PARENT_SCOPE)
        message(${ARG_MESSAGE_MODE} "Python requirement file ${ARG_REQUIREMENTS_FILE} is not installed, ${ARG_WARNING_MESSAGE}")
    else()
        set(${ARG_RESULT_VAR} ON PARENT_SCOPE)
    endif()
endfunction()
