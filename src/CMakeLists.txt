# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_subdirectory(common)

if(ENABLE_PYTHON)
    add_subdirectory(bindings/python)
endif()
