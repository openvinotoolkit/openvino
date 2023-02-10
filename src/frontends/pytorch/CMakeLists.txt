# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # 'argument': conversion from 'size_t' to 'int', possible loss of data
    ie_add_compiler_flags(/wd4267)
    ie_add_compiler_flags(/wd4244)
endif()

add_subdirectory(src)
