# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set (TARGET_NAME "dlclose_hook")

file (GLOB SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.c)

add_library(${TARGET_NAME} SHARED ${SRC})
