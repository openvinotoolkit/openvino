# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# Python tools
#

# MO

add_subdirectory(mo)

if(ENABLE_PYTHON)
    # Benchmark Tool
    add_subdirectory(benchmark_tool)

    # OpenVino Conversion Tool
    add_subdirectory(ovc)
endif()

# wheel openvino-dev

add_subdirectory(openvino_dev)
