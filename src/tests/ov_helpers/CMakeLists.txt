# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_subdirectory(ov_models)
add_subdirectory(ov_lpt_models)

if(TARGET openvino::snippets)
    add_subdirectory(ov_snippets_models)
endif()
