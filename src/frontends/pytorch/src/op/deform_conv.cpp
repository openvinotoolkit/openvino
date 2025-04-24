// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/deformable_convolution.hpp"
#include "pt_framework_node.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_deform_conv(const NodeContext& context) {
    // torchvision::deform_conv2d(Tensor input, Tensor weight, Tensor offset,
    //                            Tensor mask, Tensor bias, int64_t stride_h, int64_t stride_w,
    //                            int64_t pad_h, int64_t pad_w, int64_t dilation_h, int64_t dilation_w,
    //                            int64_t n_weight_grps, int64_t n_offset_grps, bool use_mask) -> Tensor
    num_inputs_check(context, 14, 14);
    auto pt_input = context.get_input(0);
    auto pt_weight = context.get_input(1);
    auto pt_offset = context.get_input(2);
    auto pt_mask = context.get_input(3);

    int32_t pt_stride_h = context.const_input<int32_t>(5);
    int32_t pt_stride_w = context.const_input<int32_t>(6);
    auto strides = Strides({(size_t)pt_stride_h, (size_t)pt_stride_w});

    int32_t pt_pad_h = context.const_input<int32_t>(7);
    int32_t pt_pad_w = context.const_input<int32_t>(8);
    auto pads = CoordinateDiff({pt_pad_h, pt_pad_w});

    int32_t pt_dilation_h = context.const_input<int32_t>(9);
    int32_t pt_dilation_w = context.const_input<int32_t>(10);
    auto dilations = Strides({(size_t)pt_dilation_h, (size_t)pt_dilation_w});

    int32_t pt_n_weight_grps = context.const_input<int32_t>(11);
    int32_t pt_n_offset_grps = context.const_input<int32_t>(12);
    bool pt_use_mask = context.const_input<bool>(13);

    std::shared_ptr<ov::Node> deformable_convolution;
    if (!pt_use_mask) {
        deformable_convolution = context.mark_node(std::make_shared<v8::DeformableConvolution>(pt_input,
                                                                                               pt_offset,
                                                                                               pt_weight,
                                                                                               strides,
                                                                                               pads,
                                                                                               pads,
                                                                                               dilations,
                                                                                               PadType::EXPLICIT,
                                                                                               pt_n_weight_grps,
                                                                                               pt_n_offset_grps,
                                                                                               true));
    } else {
        deformable_convolution = context.mark_node(std::make_shared<v8::DeformableConvolution>(pt_input,
                                                                                               pt_offset,
                                                                                               pt_weight,
                                                                                               pt_mask,
                                                                                               strides,
                                                                                               pads,
                                                                                               pads,
                                                                                               dilations,
                                                                                               PadType::EXPLICIT,
                                                                                               pt_n_weight_grps,
                                                                                               pt_n_offset_grps,
                                                                                               true));
    }

    if (!context.input_is_none(4)) {
        auto bias = context.get_input(4);
        bias = reshape_channelwise(context, bias, deformable_convolution);
        deformable_convolution = context.mark_node(std::make_shared<v1::Add>(deformable_convolution, bias));
    }
    return {context.mark_output(deformable_convolution)};
}
}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
