# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_subdirectory(c)

if(ENABLE_PYTHON)
    add_subdirectory(python)
endif()

