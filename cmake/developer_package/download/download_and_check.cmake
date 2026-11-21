# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

find_package(Wget QUIET)

# CI_DEPS_DOWNLOAD_REFERER, when set, is sent as the HTTP Referer header
# by every dependency downloader (wget/aria2c: --header=Referer:<value>,
# file(DOWNLOAD): HTTPHEADER).
function (GetDepsDownloadRefererHeader header_var)
  if(DEFINED ENV{CI_DEPS_DOWNLOAD_REFERER} AND NOT "$ENV{CI_DEPS_DOWNLOAD_REFERER}" STREQUAL "")
    set(${header_var} "Referer: $ENV{CI_DEPS_DOWNLOAD_REFERER}" PARENT_SCOPE)
  else()
    set(${header_var} "" PARENT_SCOPE)
  endif()
endfunction(GetDepsDownloadRefererHeader)

function (DownloadAndCheck from to fatal result sha256)
  set(status_res "ON")
  set(output 1)

  get_filename_component(download_dir ${to} DIRECTORY)
  if (NOT EXISTS ${download_dir})
    file(MAKE_DIRECTORY ${download_dir})
  endif()

  if(NOT EXISTS "${to}")
    if (${from} MATCHES "(http:)|(https:)|(ftp:)")
      message(STATUS "Downloading from ${from} to ${to} ...")
      find_program(aria2c "aria2c")
      if (${aria2c} STREQUAL "aria2c-NOTFOUND")
        if (NOT WGET_FOUND)
          Download(${from} ${to} ${fatal} ${result} output ${sha256})
          list(GET output 0 status_code)
        else()
          set(wget_args "--no-cache" "--retry-connrefused" "--waitretry=1" "--read-timeout=20" "--timeout=15" "--tries=5")
          GetDepsDownloadRefererHeader(referer_header)
          if(referer_header)
            list(APPEND wget_args "--header=${referer_header}")
          endif()
          string(JOIN " " wget_args_display ${wget_args})
          foreach(index RANGE 5)
            message(STATUS "${WGET_EXECUTABLE} ${wget_args_display} ${from}")
            execute_process(COMMAND ${WGET_EXECUTABLE} ${wget_args}
              "${from}" "-O" "${to}"
              TIMEOUT 2000
              RESULT_VARIABLE status_code)
            file(SHA256 ${to} CHECKSUM)
            if (${CHECKSUM} STREQUAL ${sha256})
              break()
            endif()
          endforeach()
          if (NOT ${CHECKSUM} STREQUAL ${sha256})
            message(FATAL_ERROR "Hash mismatch:\n"
              "expected: ${sha256}\n"
              "got: ${CHECKSUM}")
          endif()
        endif()
      else()
        set(aria2_args "-s10" "-x10" "--dir=${download_dir}")
        GetDepsDownloadRefererHeader(referer_header)
        if(referer_header)
          list(APPEND aria2_args "--header=${referer_header}")
        endif()
        string(JOIN " " aria2_args_display ${aria2_args})
        message(STATUS "${aria2c} ${aria2_args_display} ${from}")
        execute_process(COMMAND "${aria2c}" ${aria2_args} "${from}"
            TIMEOUT 2000
            RESULT_VARIABLE status_code)
      endif()

      if(NOT status_code EQUAL 0)
        if (fatal)
          message(FATAL_ERROR "fatal error: downloading '${from}' failed
            status_code: ${status_code}
            status_string: ${status_string}
            log: ${log}")
        else()
          set(status_res "ARCHIVE_DOWNLOAD_FAIL")
          message("error: downloading '${from}' failed
            status_code: ${status_code}")
        endif()
      endif()
    else()
      message(STATUS "Copying from local folder ${from} to ${to} ... ")
      file(COPY ${from} DESTINATION ${download_dir})
    endif()
  endif()

  file(REMOVE ${to}.md5)
  set(${result} "${status_res}" PARENT_SCOPE)

endfunction(DownloadAndCheck)
