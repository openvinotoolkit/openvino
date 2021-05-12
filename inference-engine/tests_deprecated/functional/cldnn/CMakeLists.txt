# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_executable(ClDnnFunctionalTests dummy.cpp)

target_link_libraries(ClDnnFunctionalTests PRIVATE gtest_main)
