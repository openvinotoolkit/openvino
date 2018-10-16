# Copyright (C) 2018 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

function (ie_option variable description value)
    option(${variable} "${description}" ${value})
    list (APPEND IE_OPTIONS "${variable}")

    set (IE_OPTIONS "${IE_OPTIONS}" PARENT_SCOPE)
endfunction()

include(version)

function (print_enabled_features)
    message(STATUS "CI_BUILD_NUMBER: ${CI_BUILD_NUMBER}")
    foreach(_var ${IE_OPTIONS})
        message(STATUS "${_var} = ${${_var}}")
    endforeach()
endfunction()
