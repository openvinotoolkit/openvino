# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

ie_add_sample(NAME classification_of_detected_roi
              SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
              HEADERS classification_of_detected_roi.h
              DEPENDENCIES format_reader)
