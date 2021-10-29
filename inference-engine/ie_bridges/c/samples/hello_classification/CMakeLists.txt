# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

ie_add_sample(NAME hello_classification_c
              SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/main.c"
              DEPENDENCIES opencv_c_wrapper)
