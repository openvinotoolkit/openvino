# Copyright (C) 2018 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required (VERSION 2.8)

include ("download")

function (resolve_archive_dependency VAR COMPONENT ARCHIVE ARCHIVE_UNIFIED ARCHIVE_WIN ARCHIVE_LIN ARCHIVE_MAC TARGET_PATH FOLDER ENVIRONMENT)

  if (ENVIRONMENT AND (DEFINED ENV{${ENVIRONMENT}}))
    set(HAS_ENV "TRUE")
  endif()

  if (NOT DEFINED HAS_ENV)
    if (ARCHIVE)
      #TODO: check wether this is platform specific binary with same name per or it is in common folder
      DownloadAndExtract(${COMPONENT} ${ARCHIVE} ${TARGET_PATH} result_path ${FOLDER})
    else()
      DownloadAndExtractPlatformSpecific(${COMPONENT} ${ARCHIVE_UNIFIED} ${ARCHIVE_WIN} ${ARCHIVE_LIN} ${ARCHIVE_MAC} ${TARGET_PATH} result_path  ${FOLDER})
    endif()

    set (${VAR} ${result_path} PARENT_SCOPE)
  else()
    set (${VAR} $ENV{${ENVIRONMENT}} PARENT_SCOPE)
  endif()

endfunction(resolve_archive_dependency)

function(resolve_pull_request GITHUB_PULL_REQUEST TARGET_PATH)
    get_filename_component(FILE_NAME ${GITHUB_PULL_REQUEST} NAME)
    set (PATCH_URL "")
    DownloadAndApply("${PATCH_URL}/${GITHUB_PULL_REQUEST}" "${IE_MAIN_SOURCE_DIR}/${TARGET_PATH}/${FILE_NAME}")
endfunction(resolve_pull_request)

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
  set(SUPPORTED_ARGS FOLDER ARCHIVE ARCHIVE_UNIFIED ARCHIVE_WIN ARCHIVE_LIN ARCHIVE_MAC TARGET_PATH ENVIRONMENT GITHUB_PULL_REQUEST VERSION_REGEX)


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

  if (NOT DEFINED ENVIRONMENT)
    set (ENVIRONMENT "OFF")
  endif()

  if (NOT DEFINED FOLDER)
    set (FOLDER FALSE)
  endif()

  #for each dependency type have to do separate things
  if (ARCHIVE_WIN OR ARCHIVE_LIN OR ARCHIVE_MAC OR ARCHIVE OR ARCHIVE_UNIFIED)
    if (NOT DEFINED TARGET_PATH)
      message(FATAL_ERROR "TARGET_PATH should be defined for every dependency")
    endif()

    resolve_archive_dependency(RESULT ${NAME_OF_CMAKE_VAR} ${ARCHIVE} ${ARCHIVE_UNIFIED} ${ARCHIVE_WIN} ${ARCHIVE_LIN} ${ARCHIVE_MAC} ${TARGET_PATH} ${FOLDER} ${ENVIRONMENT})
    set(${NAME_OF_CMAKE_VAR} ${RESULT} PARENT_SCOPE)
    if (VERSION_REGEX)
        GetNameAndUrlToDownload(archive RELATIVE_URL ${ARCHIVE_UNIFIED} ${ARCHIVE_WIN} ${ARCHIVE_LIN} ${ARCHIVE_MAC})
        if (archive)
            read_version(${archive} ${VERSION_REGEX} "${NAME_OF_CMAKE_VAR}_VERSION")
        endif()
    endif()

  elseif (DEFINED GITHUB_PULL_REQUEST)
    resolve_pull_request(${GITHUB_PULL_REQUEST} ${TARGET_PATH})
  else()
    message(FATAL_ERROR "Dependency of unknowntype, SHOULD set one of ARCHIVE_WIN, ARCHIVE, ARCHIVE_LIN, ARCHIVE_MAC, GITHUB_PULL_REQUEST")
  endif()

endfunction(RESOLVE_DEPENDENCY)

function (resolve_model_dependency network archive network_model_path)
  RESOLVE_DEPENDENCY(${network_model_path}
        ARCHIVE "models_archives/${archive}"
        TARGET_PATH "${MODELS_PATH}/${network}")
  string (REPLACE ${MODELS_PATH} "" relative_path ${${network_model_path}})
  set(${network_model_path} ".${relative_path}" PARENT_SCOPE)
endfunction()
