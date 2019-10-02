# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

ie_add_sample(NAME hello_classification
              SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
              OPENCV_DEPENDENCIES imgcodecs)
