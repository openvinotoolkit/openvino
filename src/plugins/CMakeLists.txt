# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(ENABLE_HETERO)
    add_subdirectory(hetero)
endif()

if(ENABLE_INTEL_CPU)
    add_subdirectory(intel_cpu)
endif()

if(ENABLE_INTEL_GPU)
    add_subdirectory(intel_gpu)
endif()

if(ENABLE_INTEL_GNA)
    add_subdirectory(intel_gna)
endif()

if(ENABLE_AUTO OR ENABLE_MULTI)
    add_subdirectory(auto)
endif()
