# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Usage: ie_option(<option_variable> "description" <initial value or boolean expression> [IF <condition>])

function (ie_option variable description value)
	list(FIND IE_OPTIONS "${variable}" result)

	if(${result} EQUAL -1)
	    option(${variable} "${description}" ${value})
	    list (APPEND IE_OPTIONS "${variable}")

	    set (IE_OPTIONS "${IE_OPTIONS}" PARENT_SCOPE)
	endif()
endfunction()

include(version)

function (print_enabled_features)
	message(STATUS "Inference Engine enabled features: ")
    message("")
    message("    CI_BUILD_NUMBER: ${CI_BUILD_NUMBER}")
    foreach(_var ${IE_OPTIONS})
        message("    ${_var} = ${${_var}}")
    endforeach()
    message("")
endfunction()
