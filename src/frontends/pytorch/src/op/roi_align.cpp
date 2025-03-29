// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/roi_align.hpp"

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/reshape.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_roi_align(const NodeContext& context) {
    num_inputs_check(context, 7, 7);
    auto const_1 = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {1}));
    auto const_neg_1 = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {-1}));
    auto const_0 = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {0}));
    auto const_rois_indices = context.mark_node(v0::Constant::create(element::i32, Shape{4}, {1, 2, 3, 4}));

    auto input = context.get_input(0);
    auto boxes_input = context.get_input(1);

    auto input_real_type = context.mark_node(std::make_shared<v0::Convert>(input, element::f32));
    auto boxes = context.mark_node(std::make_shared<v1::ConvertLike>(boxes_input, input_real_type));

    auto spatial_scale = context.const_input<float>(2);
    int output_size_h = context.const_input<int32_t>(3);
    int output_size_w = context.const_input<int32_t>(4);
    int sampling_ratio = context.const_input<int32_t>(5);

    auto aligned = context.const_input<bool>(6);

    auto rois = context.mark_node(std::make_shared<v8::Gather>(boxes, const_rois_indices, const_1));

    auto batch_indices_gather = context.mark_node(std::make_shared<v8::Gather>(boxes, const_0, const_1));
    auto batch_indices_reshape =
        context.mark_node(std::make_shared<v1::Reshape>(batch_indices_gather, const_neg_1, false));
    auto batch_indices = context.mark_node(std::make_shared<v0::Convert>(batch_indices_reshape, element::i32));

    v9::ROIAlign::AlignedMode aligned_mode =
        aligned ? v9::ROIAlign::AlignedMode::HALF_PIXEL_FOR_NN : v9::ROIAlign::AlignedMode::ASYMMETRIC;

    auto roi_align = context.mark_node(std::make_shared<v9::ROIAlign>(input_real_type,
                                                                      rois,
                                                                      batch_indices,
                                                                      output_size_h,
                                                                      output_size_w,
                                                                      sampling_ratio,
                                                                      spatial_scale,
                                                                      v9::ROIAlign::PoolingMode::AVG,
                                                                      aligned_mode));

    return {roi_align};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
