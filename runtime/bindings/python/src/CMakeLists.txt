# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_subdirectory(compatibility)

if(NGRAPH_PYTHON_BUILD_ENABLE)
    add_subdirectory(pyopenvino)
endif()
