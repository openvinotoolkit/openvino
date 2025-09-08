# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

function (DownloadAndApply URL apply_to sha256)

  if (EXISTS ${apply_to})
    file(READ ${apply_to} patchFile4Bytes LIMIT 4)
  endif()
  debug_message(STATUS "patchFile=" ${apply_to})
  debug_message(STATUS "patchFile4Bytes=" ${patchFileSize} "\"")

  if (NOT EXISTS ${apply_to} OR NOT patchFile4Bytes)
    #remove empty file
    if (EXISTS ${apply_to})
      file(REMOVE ${apply_to})
    endif()
    
    DownloadAndCheck(${URL} ${apply_to} TRUE result ${sha256})
  else ()
    set (MIGHT_BE_APPLIED 1)
  endif()

    get_filename_component(apply_dir ${apply_to} DIRECTORY)
    get_filename_component(patch_name ${apply_to} NAME)

    # git apply silently if file wasnot downloaded :
    #
    if (NOT DEFINED MIGHT_BE_APPLIED)
        debug_message(STATUS "applying patch ... ")
    endif()

if (NOT MINGW)
    execute_process(COMMAND git apply --verbose  ${patch_name}
      WORKING_DIRECTORY ${apply_dir}
      RESULT_VARIABLE rv
      ERROR_VARIABLE err)
else()
    debug_message("git patch: " ${patch_name} " not applied under MINGW, however it downloaded")
endif()

    #TODO:bad patch indicator either it is bad by itself of failed to apply
    #currently cannot detect what happened 
    if (NOT DEFINED MIGHT_BE_APPLIED AND NOT (rv EQUAL 0))
      file(REMOVE_RECURSE "${apply_to}")
      debug_message(FATAL_ERROR "cannot apply patch ${patch_name} " ${err})
    endif()
  
endfunction(DownloadAndApply)
