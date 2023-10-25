// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/prior_box.hpp"

#include "default_opset.hpp"
#include "exceptions.hpp"
#include "ngraph/node.hpp"
#include "onnx_import/core/node.hpp"
#include "op/org.openvinotoolkit/prior_box.hpp"

namespace ngraph {
namespace onnx_import {
namespace op {
namespace detail {
namespace {
std::shared_ptr<default_opset::StridedSlice> make_slice(std::shared_ptr<ngraph::Node> node,
                                                        int64_t start,
                                                        int64_t end) {
    return std::make_shared<default_opset::StridedSlice>(
        node,
        default_opset::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{start}),
        default_opset::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{end}),
        std::vector<int64_t>{0},   // begin mask
        std::vector<int64_t>{0});  // end mask
}
}  // namespace
}  // namespace detail

namespace set_1 {
OutputVector prior_box(const Node& node) {
    auto inputs = node.get_ng_inputs();
    NGRAPH_CHECK(inputs.size() == 2, "Invalid number of inputs");

    auto output_shape = std::make_shared<default_opset::ShapeOf>(inputs[0]);
    auto image_shape = std::make_shared<default_opset::ShapeOf>(inputs[1]);
    auto output_shape_slice = detail::make_slice(output_shape, 2, 4);
    auto image_shape_slice = detail::make_slice(image_shape, 2, 4);

    ngraph::op::v8::PriorBox::Attributes attrs;
    attrs.min_size = node.get_attribute_value<std::vector<float>>("min_size", {});
    attrs.max_size = node.get_attribute_value<std::vector<float>>("max_size", {});
    attrs.aspect_ratio = node.get_attribute_value<std::vector<float>>("aspect_ratio", {});
    attrs.flip = node.get_attribute_value<int64_t>("flip", 0);
    attrs.clip = node.get_attribute_value<int64_t>("clip", 0);
    attrs.step = node.get_attribute_value<float>("step", 0);
    attrs.offset = node.get_attribute_value<float>("offset", 0);
    attrs.variance = node.get_attribute_value<std::vector<float>>("variance", {});
    attrs.scale_all_sizes = node.get_attribute_value<int64_t>("scale_all_sizes", 1);
    attrs.fixed_ratio = node.get_attribute_value<std::vector<float>>("fixed_ratio", {});
    attrs.fixed_size = node.get_attribute_value<std::vector<float>>("fixed_size", {});
    attrs.density = node.get_attribute_value<std::vector<float>>("density", {});
    attrs.min_max_aspect_ratios_order = node.get_attribute_value<int64_t>("min_max_aspect_ratios_order", 1);

    auto axes = default_opset::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{0});

    return {std::make_shared<default_opset::Unsqueeze>(
        std::make_shared<ngraph::op::v8::PriorBox>(output_shape_slice, image_shape_slice, attrs),
        axes)};
}

OutputVector prior_box_clustered(const Node& node) {
    auto inputs = node.get_ng_inputs();
    NGRAPH_CHECK(inputs.size() == 2, "Invalid number of inputs");

    auto output_shape_rank = inputs[0].get_partial_shape().rank().get_length();
    auto image_shape_rank = inputs[1].get_partial_shape().rank().get_length();
    CHECK_VALID_NODE(node,
                     output_shape_rank == 4,
                     "Only 4D inputs are supported. First input rank: ",
                     output_shape_rank,
                     " (should be 4)");
    CHECK_VALID_NODE(node,
                     image_shape_rank == 4,
                     "Only 4D inputs are supported. Second input rank: ",
                     image_shape_rank,
                     " (should be 4)");

    auto output_shape = std::make_shared<default_opset::ShapeOf>(inputs[0]);
    auto image_shape = std::make_shared<default_opset::ShapeOf>(inputs[1]);
    auto output_shape_slice = detail::make_slice(output_shape, 2, 4);
    auto image_shape_slice = detail::make_slice(image_shape, 2, 4);

    default_opset::PriorBoxClustered::Attributes attrs{};
    attrs.widths = node.get_attribute_value<std::vector<float>>("width");
    attrs.heights = node.get_attribute_value<std::vector<float>>("height");
    attrs.clip = static_cast<bool>(node.get_attribute_value<int64_t>("clip", 0));
    attrs.variances = node.get_attribute_value<std::vector<float>>("variance", {0.1f});
    attrs.step_heights = node.get_attribute_value<float>("step_h", 0.0f);
    attrs.step_widths = node.get_attribute_value<float>("step_w", 0.0f);
    attrs.step = node.get_attribute_value<float>("step", 0.0f);
    attrs.offset = node.get_attribute_value<float>("offset", 0.0f);

    auto axes = default_opset::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{0});

    return {std::make_shared<default_opset::Unsqueeze>(
        std::make_shared<default_opset::PriorBoxClustered>(output_shape_slice, image_shape_slice, attrs),
        axes)};
}

}  // namespace set_1

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
