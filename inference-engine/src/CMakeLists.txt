# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_subdirectory(preprocessing)

if(ENABLE_MKL_DNN)
    add_subdirectory(mkldnn_plugin)
endif()

if(ENABLE_CLDNN)
    add_subdirectory(cldnn_engine)
endif()

if(ENABLE_VPU)
  add_subdirectory(vpu)
endif()

if (ENABLE_GNA)
    add_subdirectory(gna_plugin)
endif()

add_subdirectory(hetero_plugin)

add_subdirectory(inference_engine)
