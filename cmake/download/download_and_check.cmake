# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include (FindWget)

function (DownloadAndCheck from to fatal result)
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
        if (NOT ${WGET_FOUND})
          Download(${from} ${to} ${fatal} ${result} output)
          list(GET output 0 status_code)
        else()
          message(STATUS "${WGET_EXECUTABLE} --no-cache ${from}")
          execute_process(COMMAND ${WGET_EXECUTABLE} "--no-cache" "--no-check-certificate" "${from}" "-O" "${to}"
            TIMEOUT 2000
            RESULT_VARIABLE status_code)
        endif()
      else()
        message(STATUS "${aria2c} ,*.*.*.* -d ${download_dir} ${from}")
        execute_process(COMMAND "${aria2c}" "-s10" "-x10" "--dir=${download_dir}" "${from}"
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
