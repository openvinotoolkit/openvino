# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

ov_add_sample(NAME hello_affinity
              SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/inference.cpp"
                      "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
                      "${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp"
              HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/inference.hpp"
                      "${CMAKE_CURRENT_SOURCE_DIR}/utils.hpp"
              DEPENDENCIES ie_samples_utils)
