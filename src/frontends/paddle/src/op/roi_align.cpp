// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
using AlignedMode = default_opset::ROIAlign::AlignedMode;
using PoolingMode = default_opset::ROIAlign::PoolingMode;
NamedOutputs roi_align(const NodeContext& node) {
    const auto data_node = node.get_input("X");
    const auto roi_node = node.get_input("ROIs");
    const auto aligned = node.get_attribute("aligned", false);
    // Paddle only use 'avg' interpolation mode
    const auto pooling_mode = PoolingMode::AVG;
    AlignedMode aligned_mode;
    if (aligned)
        aligned_mode = AlignedMode::HALF_PIXEL_FOR_NN;
    else
        aligned_mode = AlignedMode::ASYMMETRIC;

    // TODO: support multiple batches #83232
    if (data_node.get_partial_shape().rank().is_static() && data_node.get_partial_shape()[0].is_static())
        PADDLE_OP_CHECK(node, data_node.get_partial_shape()[0] == 1, "roi_align currenty only support batch_size = 1!");

    const auto roi_node_shape = std::make_shared<default_opset::ShapeOf>(roi_node, element::i32);
    const auto start = default_opset::Constant::create(element::i64, {1}, {0});
    const auto stop = default_opset::Constant::create(element::i64, {1}, {1});
    const auto step = default_opset::Constant::create(element::i64, {1}, {1});
    const auto roisNum = std::make_shared<default_opset::Slice>(roi_node_shape, start, stop, step);

    const auto zero_const = std::make_shared<default_opset::Constant>(element::i32, Shape{1}, 0);
    const auto fake_roisNum_node = std::make_shared<default_opset::Broadcast>(zero_const, roisNum);

    const auto pooled_h = node.get_attribute<int>("pooled_height", 1);
    const auto pooled_w = node.get_attribute<int>("pooled_width", 1);
    const auto spatial_scale = node.get_attribute<float>("spatial_scale", 1.0);
    auto sampling_ratio = node.get_attribute<int>("sampling_ratio", -1);
    sampling_ratio = (sampling_ratio <= 0) ? 0 : sampling_ratio;

    return node.default_single_output_mapping({std::make_shared<default_opset::ROIAlign>(data_node,
                                                                                         roi_node,
                                                                                         fake_roisNum_node,
                                                                                         pooled_h,
                                                                                         pooled_w,
                                                                                         sampling_ratio,
                                                                                         spatial_scale,
                                                                                         pooling_mode,
                                                                                         aligned_mode)},
                                              {"Out"});
}
}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
