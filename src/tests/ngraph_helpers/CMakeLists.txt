# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_subdirectory(ngraph_functions)
add_subdirectory(lpt_ngraph_functions)

if(TARGET inference_engine_snippets)
    add_subdirectory(snippets_ngraph_functions)
endif()
