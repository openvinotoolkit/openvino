# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include (download/download)

function (resolve_archive_dependency VAR COMPONENT ARCHIVE ARCHIVE_UNIFIED ARCHIVE_WIN ARCHIVE_LIN ARCHIVE_MAC ARCHIVE_ANDROID TARGET_PATH FOLDER ENVIRONMENT SHA256 FILES_TO_EXTRACT USE_NEW_LOCATION)
  if (ENVIRONMENT AND (DEFINED ${ENVIRONMENT} OR DEFINED ENV{${ENVIRONMENT}}))
    set(HAS_ENV "TRUE")
  endif()

  if (NOT DEFINED HAS_ENV)
    if (ARCHIVE)
      #TODO: check whether this is platform specific binary with same name per or it is in common folder
      DownloadAndExtract(${COMPONENT} ${ARCHIVE} ${TARGET_PATH} result_path ${FOLDER} ${SHA256} ${FILES_TO_EXTRACT} ${USE_NEW_LOCATION})
    else()
      DownloadAndExtractPlatformSpecific(${COMPONENT} ${ARCHIVE_UNIFIED} ${ARCHIVE_WIN} ${ARCHIVE_LIN} ${ARCHIVE_MAC} ${ARCHIVE_ANDROID} ${TARGET_PATH} result_path ${FOLDER} ${SHA256} ${FILES_TO_EXTRACT} ${USE_NEW_LOCATION})
    endif()

    set (${VAR} ${result_path} PARENT_SCOPE)
  else()
    if (DEFINED ${ENVIRONMENT})
      set (${VAR} ${${ENVIRONMENT}} PARENT_SCOPE)
    else ()
      set (${VAR} $ENV{${ENVIRONMENT}} PARENT_SCOPE)
    endif ()
  endif()
endfunction(resolve_archive_dependency)

function(extract_version_from_filename filename regex version)
    string(REGEX MATCH ${regex} match ${filename})

    if (CMAKE_MATCH_1)
        set(${version} ${CMAKE_MATCH_1} PARENT_SCOPE)
    else()
        set(${version} ${filename} PARENT_SCOPE)
    endif()
endfunction(extract_version_from_filename)

function(read_version archive regex version_var)
    extract_version_from_filename(${archive} ${regex} version)
    set(${version_var} "${version}" CACHE INTERNAL "" FORCE)
    debug_message(STATUS "${version_var} = " ${version})
endfunction(read_version)

function (RESOLVE_DEPENDENCY NAME_OF_CMAKE_VAR)
  list(REMOVE_AT ARGV 0)
  set(SUPPORTED_ARGS FOLDER ARCHIVE ARCHIVE_UNIFIED ARCHIVE_WIN ARCHIVE_LIN ARCHIVE_MAC ARCHIVE_ANDROID TARGET_PATH ENVIRONMENT VERSION_REGEX SHA256 FILES_TO_EXTRACT USE_NEW_LOCATION)

  #unnecessary vars
  foreach(arg ${ARGV})
    #message("one_arg=" ${one_arg})
    #message("arg=" ${arg})
    #parse no arg vars
    if (";${SUPPORTED_ARGS};" MATCHES ";${arg};")
      if(DEFINED one_arg)
        set(${one_arg} TRUE)
      endif()
      set (one_arg ${arg})
    elseif(DEFINED one_arg)
      set(${one_arg} ${arg})
      unset(one_arg)
    else()
      message(FATAL_ERROR "invalid argument passed to resolve dependency: " ${arg})
    endif()
  endforeach(arg)

  #if last token was bool
  if(DEFINED one_arg)
    set(${one_arg} TRUE)
  endif()


  if (NOT DEFINED ARCHIVE)
    SET(ARCHIVE "OFF")
  endif()

  if (NOT DEFINED ARCHIVE_UNIFIED)
    SET(ARCHIVE_UNIFIED "OFF")
  endif()

  if (NOT DEFINED ARCHIVE_WIN)
    SET(ARCHIVE_WIN "OFF")
  endif()

  if (NOT DEFINED ARCHIVE_LIN)
    SET(ARCHIVE_LIN "OFF")
  endif()

  if (NOT DEFINED ARCHIVE_MAC)
    SET(ARCHIVE_MAC "OFF")
  endif()

  if (NOT DEFINED ARCHIVE_ANDROID)
    SET(ARCHIVE_ANDROID "OFF")
  endif()

  if (NOT DEFINED ENVIRONMENT)
    set (ENVIRONMENT "OFF")
  endif()

  if (NOT DEFINED FOLDER)
    set (FOLDER FALSE)
  endif()

  if (NOT DEFINED SHA256)
    message(FATAL_ERROR "SHA is not specified for: " ${NAME_OF_CMAKE_VAR})
  endif()

  if (NOT DEFINED FILES_TO_EXTRACT)
    set (FILES_TO_EXTRACT FALSE)
  endif()

  if (NOT DEFINED USE_NEW_LOCATION)
    set (USE_NEW_LOCATION FALSE)
  endif()

  #for each dependency type have to do separate things
  if (ARCHIVE_WIN OR ARCHIVE_LIN OR ARCHIVE_MAC OR ARCHIVE_ANDROID OR ARCHIVE OR ARCHIVE_UNIFIED)
    if (NOT DEFINED TARGET_PATH)
      message(FATAL_ERROR "TARGET_PATH should be defined for every dependency")
    endif()

    resolve_archive_dependency(RESULT ${NAME_OF_CMAKE_VAR} ${ARCHIVE} ${ARCHIVE_UNIFIED} ${ARCHIVE_WIN} ${ARCHIVE_LIN} ${ARCHIVE_MAC} ${ARCHIVE_ANDROID} ${TARGET_PATH} ${FOLDER} ${ENVIRONMENT} ${SHA256} ${FILES_TO_EXTRACT} ${USE_NEW_LOCATION})
    set(${NAME_OF_CMAKE_VAR} ${RESULT} PARENT_SCOPE)
    if (VERSION_REGEX)
        GetNameAndUrlToDownload(archive RELATIVE_URL ${ARCHIVE_UNIFIED} ${ARCHIVE_WIN} ${ARCHIVE_LIN} ${ARCHIVE_MAC} ${ARCHIVE_ANDROID} ${USE_NEW_LOCATION})
        if (archive)
            read_version(${archive} ${VERSION_REGEX} "${NAME_OF_CMAKE_VAR}_VERSION")
        endif()
    endif()
  else()
    message(FATAL_ERROR "Dependency of unknowntype, SHOULD set one of ARCHIVE_WIN, ARCHIVE, ARCHIVE_LIN, ARCHIVE_MAC, ARCHIVE_ANDROID")
  endif()

endfunction(RESOLVE_DEPENDENCY)

function (resolve_model_dependency network archive network_model_path)
  message(WARNING "DEPRECATED: 'resolve_model_dependency' must not be used")

  RESOLVE_DEPENDENCY(${network_model_path}
        ARCHIVE "models_archives/${archive}"
        TARGET_PATH "${MODELS_PATH}/${network}")
  string (REPLACE ${MODELS_PATH} "" relative_path ${${network_model_path}})
  set(${network_model_path} ".${relative_path}" PARENT_SCOPE)
endfunction()

function(reset_deps_cache)
    #
    # Reset the dependencies cache if it was set by dependency solver
    #
    set(need_reset FALSE)

    foreach(var_name IN LISTS ARGN)
        if(DEFINED ${var_name})
            if(${var_name} MATCHES ${TEMP})
                set(need_reset TRUE)
            endif()
        endif()
    endforeach()
    foreach(var_name IN LISTS ARGN)
        if(DEFINED ENV{${var_name}})
            if($ENV{${var_name}} MATCHES ${TEMP})
                set(need_reset TRUE)
            endif()
        endif()
    endforeach()

    if(need_reset)
        foreach(var_name IN LISTS ARGN)
            unset(${var_name} CACHE)
        endforeach()
        foreach(var_name IN LISTS ARGN)
            unset(ENV{${var_name}})
        endforeach()
    endif()
endfunction()

function(update_deps_cache VAR_NAME INTERNAL_VALUE DOC_MSG)
    #
    # Update the variable value if it wasn't provided by the user
    #

    if(NOT DEFINED ${VAR_NAME} AND NOT DEFINED ENV{${VAR_NAME}})
        # User didn't provide its own value, use INTERNAL_VALUE
        set(${VAR_NAME} ${INTERNAL_VALUE} CACHE PATH ${DOC_MSG})
    else()
        # The variable was provided by the user, don't use INTERNAL_VALUE
        if(NOT DEFINED ${VAR_NAME} AND DEFINED ENV{${VAR_NAME}})
            # User provided the variable via environment, convert it to the CACHE variable
            set(${VAR_NAME} $ENV{${VAR_NAME}} CACHE PATH ${DOC_MSG})
        endif()
    endif()
endfunction()
