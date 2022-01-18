# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

function (debug_message)
    if (VERBOSE_BUILD)
        message(${ARGV})
    endif()
endfunction()

function(clean_message type)
  string (REPLACE ";" "" output_string "${ARGN}")
  execute_process(COMMAND ${CMAKE_COMMAND} -E echo "${output_string}")
  if(${ARGV0} STREQUAL "FATAL_ERROR")
    message (FATAL_ERROR)
  endif()  
endfunction()
