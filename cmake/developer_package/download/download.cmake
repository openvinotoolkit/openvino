# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

function (Download from to fatal result output sha256)

  if((NOT EXISTS "${to}"))
    message(STATUS "Downloading from ${from} to ${to} ...")
    file(DOWNLOAD ${from} ${to}
      TIMEOUT 3600
      LOG log
      STATUS status
      SHOW_PROGRESS
      EXPECTED_HASH SHA256=${sha256})

    set (${output} ${status} PARENT_SCOPE)
  else()
    set (${output} 0 PARENT_SCOPE)
  endif()
  set(${result} "ON" PARENT_SCOPE)

endfunction(Download)

include(download/download_and_apply)
include(download/download_and_extract)
