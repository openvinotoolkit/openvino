# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_subdirectory(onnx_common)
add_subdirectory(frontend)

if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()
