# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(OV_NPU_SINGLE_IMAGE_TEST_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/argument_parse_helpers.cpp
    ${CMAKE_CURRENT_LIST_DIR}/argument_parse_helpers.hpp
    ${CMAKE_CURRENT_LIST_DIR}/image_quality_helper.cpp
    ${CMAKE_CURRENT_LIST_DIR}/image_quality_helper.hpp
    ${CMAKE_CURRENT_LIST_DIR}/main.cpp
    ${CMAKE_CURRENT_LIST_DIR}/map_metric_helpers.cpp
    ${CMAKE_CURRENT_LIST_DIR}/map_metric_helpers.hpp
    ${CMAKE_CURRENT_LIST_DIR}/semantic_segmentation_helpers.cpp
    ${CMAKE_CURRENT_LIST_DIR}/semantic_segmentation_helpers.hpp
    ${CMAKE_CURRENT_LIST_DIR}/yolo_helpers.cpp
    ${CMAKE_CURRENT_LIST_DIR}/yolo_helpers.hpp
)
