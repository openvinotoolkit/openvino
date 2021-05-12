# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_executable(MklDnnFunctionalTests dummy.cpp)

target_link_libraries(MklDnnFunctionalTests PRIVATE gtest_main)
