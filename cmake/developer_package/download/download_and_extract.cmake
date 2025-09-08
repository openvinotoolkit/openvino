# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include(download/extract)
include(download/download_and_check)

function (GetNameAndUrlToDownload name url archive_name_unified archive_name_win archive_name_lin archive_name_mac archive_name_android USE_NEW_LOCATION)
  if (archive_name_unified)
      if (USE_NEW_LOCATION)
        set (${url} "${archive_name_unified}" PARENT_SCOPE)
      else()
        set (${url} "thirdparty/unified/${archive_name_unified}" PARENT_SCOPE)
      endif()
    set (${name} ${archive_name_unified} PARENT_SCOPE)
  else()
    if(archive_name_lin)
      set (PLATFORM_FOLDER linux)
      set (archive_name ${archive_name_lin})
    elseif(archive_name_mac)
      set (PLATFORM_FOLDER mac)
      set (archive_name ${archive_name_mac})
    elseif(archive_name_android)
      set (PLATFORM_FOLDER android)
      set (archive_name ${archive_name_android})
    elseif(archive_name_win)
      set (PLATFORM_FOLDER windows)
      set (archive_name ${archive_name_win})
    else()
      return()
    endif()

    set (${name} ${archive_name} PARENT_SCOPE)
    set (${url}  "thirdparty/${PLATFORM_FOLDER}/${archive_name}" PARENT_SCOPE)
  endif()
endfunction(GetNameAndUrlToDownload)

#download from paltform specific folder from share server
function (DownloadAndExtractPlatformSpecific
  component
  archive_name_unified
  archive_name_win
  archive_name_lin
  archive_name_mac
  archive_name_android
  unpacked_path
  result_path
  folder
  sha256
  files_to_extract
  USE_NEW_LOCATION)

  GetNameAndUrlToDownload(archive_name RELATIVE_URL ${archive_name_unified} ${archive_name_win} ${archive_name_lin} ${archive_name_mac} ${archive_name_android} ${USE_NEW_LOCATION})
  if (NOT archive_name OR NOT RELATIVE_URL)
    return()
  endif()
  CheckOrDownloadAndExtract(${component} ${RELATIVE_URL} ${archive_name} ${unpacked_path} result_path2 ${folder} TRUE FALSE TRUE ${sha256} ${files_to_extract} ${USE_NEW_LOCATION})
  set (${result_path} ${result_path2} PARENT_SCOPE)

endfunction(DownloadAndExtractPlatformSpecific)

#download from common folder
function (DownloadAndExtract component archive_name unpacked_path result_path folder sha256 files_to_extract USE_NEW_LOCATION)
  set (RELATIVE_URL  "${archive_name}")
  set(fattal TRUE)
  CheckOrDownloadAndExtract(${component} ${RELATIVE_URL} ${archive_name} ${unpacked_path} result_path2 ${folder} ${fattal} result TRUE ${sha256} ${files_to_extract} ${USE_NEW_LOCATION})

  if (NOT ${result})
    DownloadAndExtractPlatformSpecific(${component} ${archive_name} ${archive_name} ${archive_name} ${unpacked_path} ${result_path2} ${folder} ${sha256} ${files_to_extract} ${USE_NEW_LOCATION})
  endif()

  set (${result_path} ${result_path2} PARENT_SCOPE)

endfunction(DownloadAndExtract)


function (DownloadAndExtractInternal URL archive_path  unpacked_path folder fattal resultExt sha256 files_to_extract)
  set (status "ON")
  DownloadAndCheck(${URL} ${archive_path} ${fattal} result1 ${sha256})
  if ("${result1}" STREQUAL "ARCHIVE_DOWNLOAD_FAIL")
    #check alternative url as well
    set (status "OFF")
    file(REMOVE_RECURSE "${archive_path}")
  endif()

  if ("${result1}" STREQUAL "CHECKSUM_DOWNLOAD_FAIL" OR "${result1}" STREQUAL "HASH_MISMATCH")
    set(status FALSE)
    file(REMOVE_RECURSE "${archive_path}")
  endif()

  if("${status}" STREQUAL "ON")
    ExtractWithVersion(${URL} ${archive_path} ${unpacked_path} ${folder} result ${files_to_extract})
  endif()

  set (${resultExt} ${status} PARENT_SCOPE)

endfunction(DownloadAndExtractInternal)

function (ExtractWithVersion URL archive_path unpacked_path folder result files_to_extract)

  debug_message("ExtractWithVersion : ${archive_path} : ${unpacked_path} : ${folder} : ${files_to_extract}")
  extract(${archive_path} ${unpacked_path} ${folder} ${files_to_extract} status)
  #dont need archive actually after unpacking
  file(REMOVE_RECURSE "${archive_path}")
  if (${status})
    set (version_file ${unpacked_path}/ie_dependency.info)
    file(WRITE ${version_file} ${URL})
  else()
    file(REMOVE_RECURSE "${unpacked_path}")
    message(FATAL_ERROR "Failed to extract the archive from ${URL}, archive ${archive_path} to folder ${unpacked_path}")
  endif()
  set (${result} ${status} PARENT_SCOPE)
endfunction (ExtractWithVersion)

function (DownloadOrExtractInternal URL archive_path unpacked_path folder fattal resultExt sha256 files_to_extract)
  debug_message("checking wether archive downloaded : ${archive_path}")
  set (downloadStatus "NOTOK")
  if (NOT EXISTS ${archive_path})
    DownloadAndExtractInternal(${URL} ${archive_path} ${unpacked_path} ${folder} ${fattal} result ${sha256} ${files_to_extract})
    if (${result})
      set (downloadStatus "OK")
    endif()
  else()

    if (ENABLE_UNSAFE_LOCATIONS)
      ExtractWithVersion(${URL} ${archive_path} ${unpacked_path} ${folder} result ${files_to_extract})
      if(NOT ${result})
        DownloadAndExtractInternal(${URL} ${archive_path} ${unpacked_path} ${folder} ${fattal} result ${sha256} ${files_to_extract})
        if (${result})
          set (downloadStatus "OK")
        endif()
      else()
        set (downloadStatus "OK")
      endif()
    else()
      debug_message("archive found on FS : ${archive_path}, however we cannot check it's checksum and think that it is invalid")
      file(REMOVE_RECURSE "${archive_path}")
      DownloadAndExtractInternal(${URL} ${archive_path} ${unpacked_path} ${folder} ${fattal} result ${sha256} ${files_to_extract})
      if (${result})
        set (downloadStatus "OK")
      endif()
    endif()

  endif()

  if (NOT ${downloadStatus} STREQUAL "OK")
    message(FATAL_ERROR "Failed to download and extract the archive from ${URL}, archive ${archive_path} to folder ${unpacked_path}")
  endif()

  if (NOT ${result})
    message(FATAL_ERROR "error: extract of '${archive_path}' failed")
  endif()

endfunction(DownloadOrExtractInternal)

function (CheckOrDownloadAndExtract component RELATIVE_URL archive_name unpacked_path result_path folder fattal resultExt use_alternatives sha256 files_to_extract USE_NEW_LOCATION)
  set (archive_path ${TEMP}/download/${archive_name})
  set (status "ON")

  if(DEFINED IE_PATH_TO_DEPS)
    set(URL "${IE_PATH_TO_DEPS}/${RELATIVE_URL}")
  elseif(DEFINED ENV{IE_PATH_TO_DEPS})
    set(URL "$ENV{IE_PATH_TO_DEPS}/${RELATIVE_URL}")
  elseif(USE_NEW_LOCATION)
    set(URL "https://storage.openvinotoolkit.org/dependencies/${RELATIVE_URL}")
  else()
    set(URL "https://download.01.org/opencv/master/openvinotoolkit/${RELATIVE_URL}")
  endif()

  #no message on recursive calls
  if (${use_alternatives})
    set(DEP_INFO "${component}=${URL}")
    debug_message (STATUS "DEPENDENCY_URL: ${DEP_INFO}")
  endif()

  debug_message ("checking that unpacked directory exist: ${unpacked_path}")

  if (NOT EXISTS ${unpacked_path})
    DownloadOrExtractInternal(${URL} ${archive_path} ${unpacked_path} ${folder} ${fattal} status ${sha256} ${files_to_extract})
  else(NOT EXISTS ${unpacked_path})
    #path exists, so we would like to check what was unpacked version
    set (version_file ${unpacked_path}/ie_dependency.info)

    if (NOT EXISTS ${version_file})
      clean_message(FATAL_ERROR "error: Dependency doesn't contain version file. Please select actions: \n"
        "if you are not sure about your FS dependency - remove it : \n"
        "\trm -rf ${unpacked_path}\n"
        "and rerun cmake.\n"
        "If your dependency is fine, then execute:\n\techo ${URL} > ${unpacked_path}/ie_dependency.info\n")
#     file(REMOVE_RECURSE "${unpacked_path}")
#     DownloadOrExtractInternal(${URL} ${archive_path} ${unpacked_path} ${fattal} status)
    else()
      if (EXISTS ${version_file})
        file(READ "${version_file}" dependency_url)
        string(REGEX REPLACE "\n" ";" dependency_url "${dependency_url}")
        #we have decided to stick each dependency to unique url that will be that record in version file
        debug_message("dependency_info on FS : \"${dependency_url}\"\n"
                      "compare to            : \"${URL}\"" )
      else ()
        debug_message("no version file available at ${version_file}")
      endif()

    if (NOT EXISTS ${version_file} OR NOT ${dependency_url} STREQUAL ${URL})
      if (${use_alternatives} AND ALTERNATIVE_PATH)
        #creating alternative_path
        string(REPLACE ${TEMP} ${ALTERNATIVE_PATH} unpacked_path ${unpacked_path})
        string(REPLACE ${TEMP} ${ALTERNATIVE_PATH} archive_path ${archive_path})

        debug_message("dependency different: use local path for fetching updated version: ${alternative_path}")
        CheckOrDownloadAndExtract(${component} ${RELATIVE_URL} ${archive_name} ${unpacked_path} ${result_path} ${folder} ${fattal} ${resultExt} FALSE ${sha256} ${files_to_extract})

      else()
        debug_message("dependency updated: download it again")
        file(REMOVE_RECURSE "${unpacked_path}")
        DownloadOrExtractInternal(${URL} ${archive_path} ${unpacked_path} ${folder} ${fattal} status ${sha256} ${files_to_extract})
      endif()
    endif ()
   endif()
  endif()

  if (${use_alternatives})
    set (${resultExt} "${status}" PARENT_SCOPE)
    set (${result_path} ${unpacked_path} PARENT_SCOPE)
  endif()



endfunction(CheckOrDownloadAndExtract)
