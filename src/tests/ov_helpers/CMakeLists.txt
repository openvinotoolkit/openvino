# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_subdirectory(ov_models)
add_subdirectory(lpt_ov_models)

if(TARGET openvino::snippets)
    add_subdirectory(snippets_ov_models)
endif()
