# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(CMAKE_COMPILER_IS_GNUCXX)
    ie_add_compiler_flags(-Wall)
    ie_add_compiler_flags(-Wmissing-declarations)
endif()

add_subdirectory(transformations)

add_subdirectory(low_precision_transformations)

add_subdirectory(offline_transformations)

if(ENABLE_MKL_DNN)
    add_subdirectory(mkldnn_plugin)
endif()

if(ENABLE_CLDNN)
    add_subdirectory(cldnn_engine)
endif()

if(ENABLE_VPU)
    add_subdirectory(vpu)
endif()

if(ENABLE_GNA)
    add_subdirectory(gna_plugin)
endif()

if(ENABLE_HETERO)
    add_subdirectory(hetero_plugin)
endif()

if(ENABLE_MULTI)
    add_subdirectory(multi_device)
endif()

# add a custom target to build all Inference Engine Core libraries

add_custom_target(ie_libraries ALL
                  DEPENDS inference_engine_transformations
                          inference_engine_lp_transformations)
