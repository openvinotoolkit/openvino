# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

ie_add_sample(NAME object_detection_sample_ssd
              SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
              HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/object_detection_sample_ssd.h"
              DEPENDENCIES format_reader ie_samples_utils)

find_package(ngraph REQUIRED)
target_link_libraries(object_detection_sample_ssd PRIVATE ${NGRAPH_LIBRARIES})
