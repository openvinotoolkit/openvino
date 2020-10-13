# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
function(ie_gen_default_header gen_header)
    file(WRITE "${gen_header}" "#define GraphGen(...) 1")
endfunction()

function(ie_gen_header gen_header itt_traces)
    include(FindPythonInterp)
	if (PYTHONINTERP_FOUND)
        execute_process(
                        COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/cc/generate_cc_header.py -t ${itt_traces} -o ${gen_header}
                        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                        RESULT_VARIABLE py_result
                        OUTPUT_VARIABLE py_output
                        ERROR_VARIABLE py_error
                    )
        if (${py_result} GREATER 0)
            message(FATAL_ERROR "Python ${py_cmd} result code ${py_result}, error: ${py_error}")
        else()
            message(STATUS "Python ${py_cmd} result code ${py_result}, output: ${py_output}")
        endif()
    else()
		message(FATAL_ERROR "Unable to locate Python Interpreter.")
	endif()
endfunction()

function(ie_forced_include target gen_header compression_mode)
    if(MSVC)
        set(forced_include "/FI ${gen_header}")
    else()
        set(forced_include "-imacros ${gen_header}")
        if(compression_mode)
            set(forced_include "${forced_include} -Wno-undef")
        endif()
    endif()
    set_target_properties(${target} PROPERTIES COMPILE_FLAGS ${forced_include})
endfunction()
