# Copyright (C) 2018 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required (VERSION 2.8)

function (Download from to fatal result output)

  if((NOT EXISTS "${to}"))
    message(STATUS "Downloading from ${from} to ${to} ...")
    file(DOWNLOAD ${from} ${to}
      TIMEOUT 3600
      LOG log
      STATUS status
      SHOW_PROGRESS)

    set (${output} ${status} PARENT_SCOPE)
  else()
    set (${output} 0 PARENT_SCOPE)
  endif()
  set(${result} "ON" PARENT_SCOPE)

endfunction(Download)

include ("download_and_apply")
include ("download_and_extract")
