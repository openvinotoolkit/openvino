# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_subdirectory(shared)

if (ENABLE_INTEL_GPU)
    add_subdirectory(gpu)
endif()

if (ENABLE_INTEL_MYRIAD)
    add_subdirectory(myriad)
endif()

add_subdirectory(conformance)
