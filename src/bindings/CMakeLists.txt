# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_subdirectory(c)

if(ENABLE_PYTHON)
    add_subdirectory(python)
endif()

