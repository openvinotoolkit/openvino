# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

ov_add_sample(NAME hello_classification_c
              SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/main.c;${CMAKE_CURRENT_SOURCE_DIR}/infer_result_util.c"
              HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/infer_result_util.h"
              DEPENDENCIES opencv_c_wrapper)
