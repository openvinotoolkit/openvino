# Copyright (C) 2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(ENABLE_CPPCHECK)
    find_program(CPPCHECK_EXECUTABLE cppcheck)

    if(NOT CPPCHECK_EXECUTABLE)
        message(WARNING "cppcheck was not found : disable static analysis")
        set(ENABLE_CPPCHECK OFF)
    endif()
endif()

function(add_cppcheck)
    if(NOT ENABLE_CPPCHECK)
        return()
    endif()

    set_property(
        TARGET ${ARGN}
        PROPERTY CXX_CPPCHECK
            ${CPPCHECK_EXECUTABLE}
            "--suppress=*:*/temp/*"
            "--suppress=*:*/thirdparty/*"
            "--error-exitcode=1"
            "--template={file}:{line}: error: [cppcheck:{severity}] {message}"
            "--quiet")
endfunction()
