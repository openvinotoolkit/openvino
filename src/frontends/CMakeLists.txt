# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_subdirectory(common)

if(OV_ONNX_FRONTEND_ENABLE)
    add_subdirectory(onnx)
endif()

if(OV_PDPD_FRONTEND_ENABLE)
    add_subdirectory(paddlepaddle)
endif()

if(OV_IR_FRONTEND_ENABLE)
    add_subdirectory(ir)
endif()

if(OV_TF_FRONTEND_ENABLE)
    add_subdirectory(tensorflow)
endif()

# used for static build
ov_generate_frontends_hpp()
