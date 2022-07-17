# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

ie_add_sample(NAME hello_classification_c_legacy
    SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/main.c"
    INCLUDE_DIRECTORIES "${OpenVINO_C_API_SOURCE_DIR}/include"
    DEPENDENCIES opencv_c_wrapper openvino_c)
