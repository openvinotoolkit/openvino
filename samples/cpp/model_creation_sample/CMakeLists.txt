# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "model_creation_sample")

ov_add_sample(NAME model_creation_sample
              SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
              HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/model_creation_sample.hpp"
              DEPENDENCIES format_reader ie_samples_utils)
