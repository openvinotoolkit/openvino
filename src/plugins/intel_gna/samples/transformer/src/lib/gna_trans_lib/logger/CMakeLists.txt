# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET gna_trans_lib_logger)
add_library(${TARGET} INTERFACE)
target_include_directories(${TARGET} INTERFACE .)

add_clang_format_target(${TARGET}_clang FOR_TARGETS ${TARGET})
