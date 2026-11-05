# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

function (Download from to fatal result output sha256)

  if((NOT EXISTS "${to}"))
    message(STATUS "Downloading from ${from} to ${to} ...")
    GetDepsDownloadRefererHeader(referer_header)
    if(referer_header)
      file(DOWNLOAD ${from} ${to}
        TIMEOUT 3600
        LOG log
        STATUS status
        SHOW_PROGRESS
        EXPECTED_HASH SHA256=${sha256}
        HTTPHEADER "${referer_header}")
    else()
      file(DOWNLOAD ${from} ${to}
        TIMEOUT 3600
        LOG log
        STATUS status
        SHOW_PROGRESS
        EXPECTED_HASH SHA256=${sha256})
    endif()

    set (${output} ${status} PARENT_SCOPE)
  else()
    set (${output} 0 PARENT_SCOPE)
  endif()
  set(${result} "ON" PARENT_SCOPE)

endfunction(Download)

include(download/download_and_apply)
include(download/download_and_extract)
