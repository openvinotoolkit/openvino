# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(GFLAGS_IS_SUBPROJECT TRUE)
set(HAVE_SYS_STAT_H 1)
set(HAVE_INTTYPES_H 1)
set(INTTYPES_FORMAT C99)
set(BUILD_TESTING OFF)
set(BUILD_SHARED_LIBS OFF)

add_subdirectory(gflags EXCLUDE_FROM_ALL)
set_target_properties(gflags_nothreads_static PROPERTIES FOLDER thirdparty)
