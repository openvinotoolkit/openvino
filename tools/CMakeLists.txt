# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# Python tools
#

if(ENABLE_PYTHON)
    # Benchmark Tool
    add_subdirectory(benchmark_tool)

    # OpenVino Conversion Tool
    add_subdirectory(ovc)
endif()
