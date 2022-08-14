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
        RESULT_VAR              # Result varibale to set return code {TRUE | FALSE}
        )
    set(oneValueOptionalArgs
        MESSAGE_MODE            # Set the type of message: { FATAL_ERROR | WARNING | ... }
        )
    set(multiValueArgs)
    
    cmake_parse_arguments(ARG "${options}" "${oneValueRequiredArgs};${oneValueOptionalArgs}" "${multiValueArgs}" ${ARGN})
    
    foreach(argName ${oneValueRequiredArgs})
        if (NOT ARG_${argName})
            message(SEND_ERROR "Argument '${argName}' is required.")
        endif()
    endforeach()
    
    if(NOT ARG_MESSAGE_MODE)
        set(ARG_MESSAGE_MODE WARNING)
    endif()
    
    if (ARG_UNPARSED_ARGUMENTS)
        message(SEND_ERROR "Unexpected parameters have passed to the function: ${ARG_UNPARSED_ARGUMENTS}")
    endif()
  
    get_filename_component(PYTHON_EXEC_DIR ${PYTHON_EXECUTABLE} DIRECTORY)
    
    STRING(REPLACE "'" "\\'" REQ "${ARG_REQUIREMENT}")
    execute_process(
        COMMAND ${PYTHON_EXECUTABLE} -c "import pkg_resources ; pkg_resources.require('${REQ}')"
        WORKING_DIRECTORY ${PYTHON_EXEC_DIR}
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
#                       [FAIL_FAST])
#
function(ov_check_pip_packages)
    find_package(PythonInterp 3 REQUIRED)
 
    set(options
        FAIL_FAST               # Exit function at first requirement failure
        )
    set(oneValueRequiredArgs
        REQUIREMENTS_FILE       # File with requirement-specifiers to check 
        RESULT_VAR              # Result varibale to set return code {TRUE | FALSE}
        )
    set(multiValueArgs)
    
    cmake_parse_arguments(ARG "${options}" "${oneValueRequiredArgs}" "${multiValueArgs}" ${ARGN})
    
    foreach(argName ${oneValueRequiredArgs})
        if (NOT ARG_${argName})
            message(SEND_ERROR "Argument '${argName}' is required.")
        endif()
    endforeach()
    
    if(ARG_UNPARSED_ARGUMENTS)
        message(SEND_ERROR "Unexpected parameters have passed to the function: ${ARG_UNPARSED_ARGUMENTS}")
    endif()
   
    set(all_reqs_found ON)
    file(STRINGS ${ARG_REQUIREMENTS_FILE} REQS)

    foreach(REQ IN LISTS REQS)
        if(REQ)
            ov_check_pip_package(REQUIREMENT ${REQ}
                                 MESSAGE_MODE WARNING
                                 RESULT_VAR req_FOUND)
            if(NOT req_FOUND)
                set(all_reqs_found OFF)
                if(ARG_FAIL_FAST)
                    message(WARNING "Dependencies are not installed or have conflicts. Please use \"${PYTHON_EXECUTABLE} -m pip install -r ${ARG_REQUIREMENTS_FILE}\".")
                    set(${ARG_RESULT_VAR} ${all_reqs_found} PARENT_SCOPE)
                    return()
                endif()
            endif()
        endif()
    endforeach()

    set(${ARG_RESULT_VAR} ${all_reqs_found} PARENT_SCOPE)
endfunction()
