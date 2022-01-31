# Copyright (C) 2018-2022 Intel Corporation
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

if(ENABLE_INTEL_MYRIAD_COMMON)
    add_subdirectory(intel_myriad)
endif()

if(ENABLE_AUTO OR ENABLE_MULTI)
    add_subdirectory(auto)
endif()

if(ENABLE_AUTO_BATCH)
    add_subdirectory(auto_batch)
endif()
