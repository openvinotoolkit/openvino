# Copyright (C) 2018-2019 Intel Corporation
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

file(REMOVE ${CMAKE_BINARY_DIR}/ld_library_rpath_64.txt)

# log relative path to shared library that has to be used in LD_LIBRARY_PATH
function (log_rpath_remove_top component component_remove_top lib lib_remove_top)
  
  set(top_lib_dir ${${component}})
  set(lib_dir ${lib})

#  debug_message(STATUS "LIB-IN=${lib} ")
#  debug_message(STATUS "TOPLIB-IN=${top_lib_dir} ")
  get_filename_component(top_lib_dir ${${component}} DIRECTORY)

  if (${component_remove_top} AND ${component})
  else()
    get_filename_component(add_name ${${component}} NAME)
    set(top_lib_dir "${top_lib_dir}/${add_name}")
  endif()
  if (${lib_remove_top} AND lib)
    get_filename_component(lib_dir ${lib} DIRECTORY)
  endif()

  string (REPLACE "//" "/" top_lib_dir "${top_lib_dir}")
  string (REPLACE "//" "/" lib_dir "${lib_dir}")

  string (REPLACE "\\\\" "/" top_lib_dir "${top_lib_dir}")
  string (REPLACE "\\\\" "/" lib_dir "${lib_dir}")

#  debug_message(STATUS "LIB-OUT=${lib_dir}")
#  debug_message(STATUS "TOPLIB-OUT=${top_lib_dir}")

  if (WIN32)
    string (TOLOWER "${top_lib_dir}" top_lib_dir)
    string (TOLOWER "${lib_dir}" lib_dir)
  endif()

  string (REPLACE "${top_lib_dir}" "" component_dir "${lib_dir}")

  set(RPATH_INFO "${component}=${component_dir}")
  debug_message(STATUS "LD_LIBRARY_RPATH: ${RPATH_INFO}")
  file(APPEND ${CMAKE_BINARY_DIR}/ld_library_rpath_64.txt "${RPATH_INFO}\n")
endfunction()

function (log_rpath_from_dir component lib_dir)
  log_rpath_remove_top("${component}" TRUE "${lib_dir}" FALSE)
endfunction()

function (log_rpath component lib_path)
  log_rpath_remove_top(${component} TRUE ${lib_path} TRUE)
endfunction()

# Just wrapping of the original message() function to make this macro known during IE build.
# This macro is redefined (with additional checks) within the InferenceEngineConfig.cmake file.
macro(ext_message TRACE_LEVEL)
    message(${TRACE_LEVEL} "${ARGN}")
endmacro()