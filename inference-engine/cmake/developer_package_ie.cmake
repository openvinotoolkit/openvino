# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# environment variables used
# name of environment variable stored path to temp directory"
set(DL_SDK_TEMP  "DL_SDK_TEMP")

# prepare temporary folder
function(set_temp_directory temp_variable source_tree_dir)
    if (DEFINED ENV{${DL_SDK_TEMP}} AND NOT $ENV{${DL_SDK_TEMP}} STREQUAL "")
        if (WIN32)
            string(REPLACE "\\" "\\\\" temp $ENV{${DL_SDK_TEMP}})
        else(WIN32)
            set(temp $ENV{${DL_SDK_TEMP}})
        endif(WIN32)

        if (ENABLE_ALTERNATIVE_TEMP)
            set(ALTERNATIVE_PATH ${source_tree_dir}/temp)
        endif()
    else ()
        message(STATUS "DL_SDK_TEMP envionment not set")
        set(temp ${source_tree_dir}/temp)
    endif()

    set("${temp_variable}" "${temp}" PARENT_SCOPE)
    if(ALTERNATIVE_PATH)
        set(ALTERNATIVE_PATH "${ALTERNATIVE_PATH}" PARENT_SCOPE)
    endif()
endfunction()

include(sanitizer)
include(cpplint)
include(cppcheck)

if(ENABLE_PROFILING_ITT)
    find_package(ITT REQUIRED)
endif()

include(plugins/plugins)
