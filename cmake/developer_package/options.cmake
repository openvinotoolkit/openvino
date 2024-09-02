# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include (CMakeDependentOption)

if(POLICY CMP0127)
    cmake_policy(SET CMP0127 NEW)
endif()

unset(OV_OPTIONS CACHE)

macro(ov_option variable description value)
    option(${variable} "${description}" ${value})
    list(APPEND OV_OPTIONS ${variable})
    set(OV_OPTIONS "${OV_OPTIONS}" CACHE INTERNAL "A list of OpenVINO cmake options")
endmacro()

macro(ov_dependent_option variable description def_value condition fallback_value)
    cmake_dependent_option(${variable} "${description}" ${def_value} "${condition}" ${fallback_value})
    list(APPEND OV_OPTIONS ${variable})
    set(OV_OPTIONS "${OV_OPTIONS}" CACHE INTERNAL "A list of OpenVINO cmake options")
endmacro()

macro(ov_option_enum variable description value)
    set(OPTIONS)
    set(ONE_VALUE_ARGS)
    set(MULTI_VALUE_ARGS ALLOWED_VALUES)
    cmake_parse_arguments(OPTION_ENUM "${OPTIONS}" "${ONE_VALUE_ARGS}" "${MULTI_VALUE_ARGS}" ${ARGN})

    if(NOT ${value} IN_LIST OPTION_ENUM_ALLOWED_VALUES)
        message(FATAL_ERROR "Internal error: variable must be one of ${OPTION_ENUM_ALLOWED_VALUES}")
    endif()

    list(APPEND OV_OPTIONS ${variable})
    set(OV_OPTIONS "${OV_OPTIONS}" CACHE INTERNAL "A list of OpenVINO cmake options")

    set(${variable} ${value} CACHE STRING "${description}")
    set_property(CACHE ${variable} PROPERTY STRINGS ${OPTION_ENUM_ALLOWED_VALUES})

    unset(OPTIONS)
    unset(ONE_VALUE_ARGS)
    unset(MULTI_VALUE_ARGS)
    unset(OPTION_ENUM_ALLOWED_VALUES)
endmacro()

function (ov_print_enabled_features)
    if(NOT COMMAND ov_set_ci_build_number)
        message(FATAL_ERROR "CI_BUILD_NUMBER is not set yet")
    endif()

    message(STATUS "OpenVINO Runtime enabled features: ")
    message(STATUS "")
    message(STATUS "    CI_BUILD_NUMBER: ${CI_BUILD_NUMBER}")
    foreach(_var IN LISTS OV_OPTIONS)
        message(STATUS "    ${_var} = ${${_var}}")
    endforeach()
    message(STATUS "")
endfunction()
