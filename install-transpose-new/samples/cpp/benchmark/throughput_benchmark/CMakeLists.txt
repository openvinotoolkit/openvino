# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

ov_add_sample(NAME throughput_benchmark
              SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
              DEPENDENCIES ie_samples_utils)
