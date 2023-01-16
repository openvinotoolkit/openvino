# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(ENABLE_MYRIAD)
    add_subdirectory(vpu_compile)
    add_subdirectory(vpu_perfcheck)
endif()
