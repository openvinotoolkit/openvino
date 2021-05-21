# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_subdirectory(shared)

if (ENABLE_MKL_DNN)
    add_subdirectory(cpu)
endif()

if (ENABLE_CLDNN)
    add_subdirectory(gpu)
endif()

if (ENABLE_GNA)
    add_subdirectory(gna)
endif()

if (ENABLE_MYRIAD)
    add_subdirectory(myriad)
endif()

add_subdirectory(conformance)
