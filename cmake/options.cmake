# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Usage: ie_option(<option_variable> "description" <initial value or boolean expression> [IF <condition>])

include (CMakeDependentOption)
include (version)

macro (ie_option variable description value)
    option(${variable} "${description}" ${value})
    list(APPEND IE_OPTIONS ${variable})
endmacro()

macro (ie_dependent_option variable description def_value condition fallback_value)
    cmake_dependent_option(${variable} "${description}" ${def_value} "${condition}" ${fallback_value})
    list(APPEND IE_OPTIONS ${variable})
endmacro()

function (print_enabled_features)
    message(STATUS "Inference Engine enabled features: ")
    message(STATUS "")
    message(STATUS "    CI_BUILD_NUMBER: ${CI_BUILD_NUMBER}")
    foreach(_var ${IE_OPTIONS})
        message(STATUS "    ${_var} = ${${_var}}")
    endforeach()
    message(STATUS "")
endfunction()
