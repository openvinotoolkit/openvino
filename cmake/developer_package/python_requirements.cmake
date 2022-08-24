# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

#
# _ov_check_pip_installed(<output var>)
#
function(_ov_check_pip_installed var)
    find_host_program(python_pip_program
                      NAMES pip pip3
                      DOC "python3-pip program")
    if(python_pip_program)
        set(${var} ON)
    endif()
endfunction()

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
            COMMAND ${PYTHON_EXECUTABLE} -c "import pkg_resources ; pkg_resources.require(open('${ARG_REQUIREMENTS_FILE}', mode='r'))"
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

#
# ov_create_virtualenv(REQUIREMENTS_FILE <requirements.txt file>
#                      VIRTUALENV_NAME <tag / id of virtual env>
#                      OUTPUT_DEPENDENCY_FILE <output_file>
#                      RESULT_VAR <result var name>
#                      [VIRTUALENV_PYTHON_EXECUTABLE <var>]
#                      [WARNING_MESSAGE <message>]
#                      [MESSAGE_MODE <WARNING | FATAL_ERROR | TRACE>])
#
function(ov_create_virtualenv)
    if(NOT PYTHON_EXECUTABLE)
        message(FATAL_ERROR "PYTHON_EXECUTABLE must be defined")
    endif()

    set(oneValueOptionalArgs
        MESSAGE_MODE               # Set the type of message: { FATAL_ERROR | WARNING | ... }
        WARNING_MESSAGE            # callback message
        VIRTUALENV_PYTHON_EXECUTABLE
        )
    set(oneValueRequiredArgs
        REQUIREMENTS_FILE          # File with requirement-specifiers to check
        VIRTUALENV_NAME            # tag / id of virtual env
        RESULT_VAR                 # Result varibale to set return code {ON | OFF}
        OUTPUT_DEPENDENCY_FILE
        )
    set(multiValueArgs)

    cmake_parse_arguments(ARG "" "${oneValueOptionalArgs};${oneValueRequiredArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT DEFINED ARG_VIRTUALENV_PYTHON_EXECUTABLE)
        set(ARG_VIRTUALENV_PYTHON_EXECUTABLE PYTHON_EXECUTABLE)
    endif()

    set(${ARG_RESULT_VAR} OFF PARENT_SCOPE)
    set(python_executable_var "PYTHON_FOR_${ARG_VIRTUALENV_NAME}")

    # check whether the environment is already created
    if(${python_executable_var})
        set(${ARG_RESULT_VAR} ON PARENT_SCOPE)
        set(${ARG_VIRTUALENV_PYTHON_EXECUTABLE} "${${python_executable_var}}" PARENT_SCOPE)
        return()
    endif()

    # 0. check whether requirements are already satisfied in the global site-packages
    ov_check_pip_packages(REQUIREMENTS_FILE "${ARG_REQUIREMENTS_FILE}"
                          RESULT_VAR requirements_satisfied
                          # silent mode
                          MESSAGE_MODE TRACE)

    if(requirements_satisfied)
        # just use global PYTHON_EXECUTABLE
        set(${ARG_RESULT_VAR} ON PARENT_SCOPE)
        set(${ARG_VIRTUALENV_PYTHON_EXECUTABLE} "${PYTHON_EXECUTABLE}" PARENT_SCOPE)
        return()
    endif()

    # 1. check whether we have python3-pip and python3-venv
    execute_process(COMMAND "${PYTHON_EXECUTABLE}" -c "import pip"
                    RESULT_VARIABLE pip_exit_code
                    OUTPUT_VARIABLE pip_output
                    ERROR_VARIABLE pip_error)
                      
    execute_process(COMMAND "${PYTHON_EXECUTABLE}" -c "import virtualenv"
                    RESULT_VARIABLE virtualenv_exit_code
                    OUTPUT_VARIABLE virtualenv_output
                    ERROR_VARIABLE virtualenv_error)

    execute_process(COMMAND "${PYTHON_EXECUTABLE}" -c "import venv"
                    RESULT_VARIABLE venv_exit_code
                    OUTPUT_VARIABLE venv_output
                    ERROR_VARIABLE venv_error)

    if(NOT ((venv_exit_code EQUAL 0) OR (virtualenv_exit_code EQUAL 0)) OR
       NOT   (pip_exit_code EQUAL 0) OR
       # explicitly disabled
       NOT ENABLE_REQUIREMENTS_INSTALL)
        # we cannot create venv in build tree
        message(${ARG_MESSAGE_MODE} "Python requirement file ${ARG_REQUIREMENTS_FILE} is not installed, ${ARG_WARNING_MESSAGE}")
        return()
    endif()

    # 2. create virtual env
    set(virtualenvs_root "${CMAKE_BINARY_DIR}/virtual_envs")
    set(virtualenv_path "${virtualenvs_root}/${ARG_VIRTUALENV_NAME}")

    if(venv_exit_code EQUAL 0)
        execute_process(COMMAND "${PYTHON_EXECUTABLE}" -m venv "${virtualenv_path}"
                        RESULT_VARIABLE venv_result
                        OUTPUT_VARIABLE output_var
                        ERROR_VARIABLE error_var)
    elseif(virtualenv_exit_code EQUAL 0)
        execute_process(COMMAND "${PYTHON_EXECUTABLE}" -m virtualenv --no-setuptools --no-wheel "${virtualenv_path}"
                        RESULT_VARIABLE venv_result
                        OUTPUT_VARIABLE output_var
                        ERROR_VARIABLE error_var)
    endif()

    if(NOT venv_result EQUAL 0)
        message(WARNING "Internal error: failed to create virtual env in ${virtualenv_path}:\n ${output_var} \n ${error_var}")
        return()
    endif()

    # 3. install all the prerequisited here

    unset(${python_executable_var} CACHE)

    find_host_program(${python_executable_var}
                      NAMES python3 python
                      PATHS "${virtualenv_path}"
                      PATH_SUFFIXES bin
                      DOC "Python interpreter for ${ARG_VIRTUALENV_NAME} venv"
                      NO_DEFAULT_PATH)

    if(NOT ${python_executable_var})
        message(FATAL_ERROR "Internal error: failed to find python3 in ${virtualenv_path}")
    endif()

    set(output_file "${virtualenvs_root}/venv_${ARG_VIRTUALENV_NAME}.txt")
    file(REMOVE "${output_file}")

    add_custom_command(OUTPUT "${output_file}"
        COMMAND "${${python_executable_var}}" -m pip install -q -r "${ARG_REQUIREMENTS_FILE}"
        COMMAND "${CMAKE_COMMAND}" -E touch "${output_file}"
        DEPENDS "${ARG_REQUIREMENTS_FILE}"
        COMMENT "Install requirements to '${ARG_VIRTUALENV_NAME}' venv"
        VERBATIM)

    set(${ARG_RESULT_VAR} ON PARENT_SCOPE)
    set(${ARG_OUTPUT_DEPENDENCY_FILE} "${output_file}" PARENT_SCOPE)
    set(${ARG_VIRTUALENV_PYTHON_EXECUTABLE} "${${python_executable_var}}" PARENT_SCOPE)
endfunction()
