# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

ie_add_sample(NAME hello_nv12_input_classification_c_legacy
    INCLUDE_DIRECTORIES "${OpenVINO_C_API_SOURCE_DIR}/include"
    SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/main.c"
    DEPENDENCIES openvino_c)
