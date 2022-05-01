# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(WIN32)
    ie_add_sample(NAME runtime_load_c
                SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")
else()
    ie_add_sample(NAME runtime_load_c
                SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
                DEPENDENCIES dl pthread)
endif()