// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <conversion_extensions.hpp>
#include <openvino/core/extension.hpp>
#include <tensorflow_frontend/extension.hpp>

OPENVINO_CREATE_EXTENSIONS(std::vector<ov::Extension::Ptr>({
    std::make_shared<ov::frontend::ConversionExtension>("_FusedConv2D",
                                                        ov::frontend::tf::op::translate_fused_conv_2d_op),
    std::make_shared<ov::frontend::ConversionExtension>("_FusedMatMul",
                                                        ov::frontend::tf::op::translate_fused_mat_mul_op),
    std::make_shared<ov::frontend::ConversionExtension>("_FusedBatchNormEx",
                                                        ov::frontend::tf::op::translate_fused_batch_norm_op),
    std::make_shared<ov::frontend::ConversionExtension>("_FusedDepthwiseConv2dNative",
                                                        ov::frontend::tf::op::translate_depthwise_conv_2d_native_op),
    std::make_shared<ov::frontend::ConversionExtension>("_Retval", ov::frontend::tf::op::translate_retval_op),
}));
