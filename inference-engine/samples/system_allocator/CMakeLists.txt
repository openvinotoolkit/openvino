# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

ie_add_sample(NAME system_allocator
              SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
              OPENCV_DEPENDENCIES imgcodecs)
