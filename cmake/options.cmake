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

macro (ie_option_enum variable description value)
    set(OPTIONS)
    set(ONE_VALUE_ARGS)
    set(MULTI_VALUE_ARGS ALLOWED_VALUES)
    cmake_parse_arguments(IE_OPTION_ENUM "${OPTIONS}" "${ONE_VALUE_ARGS}" "${MULTI_VALUE_ARGS}" ${ARGN})

    if(NOT ${value} IN_LIST IE_OPTION_ENUM_ALLOWED_VALUES)
        message(FATAL_ERROR "variable must be one of ${IE_OPTION_ENUM_ALLOWED_VALUES}")
    endif()

    list(APPEND IE_OPTIONS ${variable})

    set(${variable} ${value} CACHE STRING "${description}")
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
