# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

ie_add_sample(NAME classification_sample_async
              SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
              HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/classification_sample_async.h"
              DEPENDENCIES format_reader ie_samples_utils)
