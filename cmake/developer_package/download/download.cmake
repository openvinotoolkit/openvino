# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

function (Download from to fatal result output sha256)

  if((NOT EXISTS "${to}"))
    message(STATUS "Downloading from ${from} to ${to} ...")
    # This helps to track where the download is happening from, for analytics purposes
    set(_download_referer_header "")
    if(DEFINED ENV{CI} AND "$ENV{CI}" STREQUAL "true")
      set(_download_referer "generic-ci-cmake")
      if(DEFINED ENV{GITHUB_ACTIONS} AND "$ENV{GITHUB_ACTIONS}" STREQUAL "true")
        set(_download_referer "generic-github-actions-cmake")
        if(DEFINED ENV{GITHUB_REPOSITORY_OWNER} AND "$ENV{GITHUB_REPOSITORY_OWNER}" STREQUAL "openvinotoolkit")
          set(_download_referer "openvino-gha-ci-cmake")
        endif()
      endif()
      set(_download_referer_header HTTPHEADER "Referer: ${_download_referer}")
    endif()
    file(DOWNLOAD ${from} ${to}
      TIMEOUT 3600
      LOG log
      STATUS status
      SHOW_PROGRESS
      EXPECTED_HASH SHA256=${sha256}
      ${_download_referer_header})

    set (${output} ${status} PARENT_SCOPE)
  else()
    set (${output} 0 PARENT_SCOPE)
  endif()
  set(${result} "ON" PARENT_SCOPE)

endfunction(Download)

include(download/download_and_apply)
include(download/download_and_extract)
