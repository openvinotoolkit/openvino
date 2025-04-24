// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/prior_box.hpp"

#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
using namespace default_opset;
using namespace element;
namespace detail {
namespace {
std::shared_ptr<StridedSlice> make_slice(const std::shared_ptr<ov::Node>& node, int64_t start, int64_t end) {
    return std::make_shared<StridedSlice>(node,
                                          Constant::create(i64, Shape{1}, std::vector<int64_t>{start}),
                                          Constant::create(i64, Shape{1}, std::vector<int64_t>{end}),
                                          std::vector<int64_t>{0},   // begin mask
                                          std::vector<int64_t>{0});  // end mask
}
}  // namespace
}  // namespace detail
NamedOutputs prior_box(const NodeContext& node) {
    auto input = node.get_input("Input");
    auto Image = node.get_input("Image");
    const auto input_shape = std::make_shared<ShapeOf>(input);
    const auto Image_shape = std::make_shared<ShapeOf>(Image);
    const auto output_shape_slice = detail::make_slice(input_shape, 2, 4);
    const auto image_shape_slice = detail::make_slice(Image_shape, 2, 4);

    PriorBox::Attributes attrs;
    attrs.min_size = node.get_attribute<std::vector<float>>("min_sizes", {});
    attrs.max_size = node.get_attribute<std::vector<float>>("max_sizes", {});
    attrs.aspect_ratio = node.get_attribute<std::vector<float>>("aspect_ratios", {1.0f});
    attrs.flip = node.get_attribute<bool>("flip", false);
    attrs.clip = node.get_attribute<bool>("clip", false);
    attrs.step = node.get_attribute<float>("step_w", 0.f);
    attrs.min_max_aspect_ratios_order = node.get_attribute<bool>("min_max_aspect_ratios_order", false);

    attrs.offset = node.get_attribute<float>("offset", 0.5f);
    attrs.variance = node.get_attribute<std::vector<float>>("variances", {0.1f, 0.1f, 0.2f, 0.2f});

    const auto ov_prior_box_node = std::make_shared<PriorBox>(output_shape_slice, image_shape_slice, attrs);

    const auto split_axis_node = Constant::create(i64, ov::Shape{}, {0});
    const auto node_prior_box_split = std::make_shared<Split>(ov_prior_box_node, split_axis_node, 2);

    const auto node_boxes_origin = node_prior_box_split->output(0);
    const auto node_variances_origin = node_prior_box_split->output(1);

    const auto out_shape =
        std::make_shared<Concat>(NodeVector{output_shape_slice, Constant::create<int64_t>(i64, {2}, {-1, 4})}, 0);

    auto node_boxes_reshape = std::make_shared<Reshape>(node_boxes_origin, out_shape, true);
    const auto node_variances_reshape = std::make_shared<Reshape>(node_variances_origin, out_shape, true);

    NamedOutputs outputs;
    outputs["Boxes"] = {node_boxes_reshape};
    outputs["Variances"] = {node_variances_reshape};
    return outputs;
}
}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
