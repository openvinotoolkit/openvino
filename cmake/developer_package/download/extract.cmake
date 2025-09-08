# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

function (extract archive_path unpacked_path folder files_to_extract result)
  # Slurped from a generated extract-TARGET.cmake file.
    get_filename_component(unpacked_dir ${unpacked_path} DIRECTORY)

    file(MAKE_DIRECTORY ${unpacked_path})

    message(STATUS "extracting...
         src='${archive_path}'
         dst='${unpacked_path}'")

    if(NOT EXISTS "${archive_path}")
      message(FATAL_ERROR "error: file to extract does not exist: '${archive_path}'")
    endif()

    # Extract it:
    #
    # in case of archive dont have top level folder lets create it
    if (${folder})
      set (unpacked_dir ${unpacked_path})
      message("unpacked_dir= ${unpacked_dir}")      
    endif()

    string(REGEX REPLACE ";" " " list_files_to_extract "${${files_to_extract}}")
    message(STATUS "extracting... [tar xfz] ${list_files_to_extract}")
    execute_process(COMMAND ${CMAKE_COMMAND} -E tar xfz ${archive_path} ${${files_to_extract}}
      WORKING_DIRECTORY ${unpacked_dir}
      RESULT_VARIABLE rv
      ERROR_VARIABLE err)

    if (NOT (rv EQUAL 0))
      message(STATUS "error: extract of '${archive_path}' failed: ${err}")
      #invalid archive
      file(REMOVE_RECURSE "${unpacked_path}")
      file(REMOVE_RECURSE "${archive_path}")
      set(${result} 0 PARENT_SCOPE)
    else()
      if (NOT (err STREQUAL ""))
         message(STATUS "${err}")
      endif()
      set(${result} 1 PARENT_SCOPE)
    endif()

endfunction (extract)
