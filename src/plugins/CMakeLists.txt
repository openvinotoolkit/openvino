# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(ENABLE_HETERO)
    add_subdirectory(hetero)
endif()

if(ENABLE_INTEL_GPU)
    add_subdirectory(intel_gpu)
endif()
