# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


if(ENABLE_PYTHON OR NGRAPH_PYTHON_BUILD_ENABLE)
    add_subdirectory(bindings/python)
endif()
