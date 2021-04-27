# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

ie_add_sample(NAME object_detection_sample_ssd_c
              SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/main.c"
              HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/object_detection_sample_ssd.h"
              		  "${CMAKE_CURRENT_SOURCE_DIR}/c_w_dirent.h"
              DEPENDENCIES opencv_c_wrapper)
