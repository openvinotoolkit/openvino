# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_subdirectory(proxy)

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

# Register proxy plugin for CPU device
ov_add_plugin(NAME PROXY
              DEVICE_NAME "CPU"
              DEFAULT_CONFIG
              FALLBACK_PRIORITY:Intel_CPU)

# Register proxy plugin for GPU device
ov_add_plugin(NAME PROXY
              DEVICE_NAME "GPU"
              DEFAULT_CONFIG
              FALLBACK_PRIORITY:Intel_GPU)
