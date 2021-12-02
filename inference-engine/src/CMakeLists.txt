# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(CMAKE_COMPILER_IS_GNUCXX)
    ie_add_compiler_flags(-Wall)
    ie_add_compiler_flags(-Wmissing-declarations)
endif()

add_subdirectory(offline_transformations)

if(ENABLE_MKL_DNN)
    add_subdirectory(mkldnn_plugin)
endif()

if(ENABLE_VPU)
    add_subdirectory(vpu)
endif()

if(ENABLE_GNA)
    add_subdirectory(gna_plugin)
endif()

if(ENABLE_MULTI)
    add_subdirectory(multi_device)
endif()

if(ENABLE_BATCH)
    add_subdirectory(auto_batch)
endif()

