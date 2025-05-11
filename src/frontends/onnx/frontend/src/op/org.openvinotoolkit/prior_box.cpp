// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/prior_box.hpp"

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/prior_box_clustered.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/unsqueeze.hpp"

using namespace ov::op;
using ov::Shape;

namespace ov {
namespace frontend {
namespace onnx {
namespace org_openvinotoolkit {
namespace detail {
namespace {
std::shared_ptr<v1::StridedSlice> make_slice(std::shared_ptr<ov::Node> node, int64_t start, int64_t end) {
    return std::make_shared<v1::StridedSlice>(
        node,
        v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{start}),
        v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{end}),
        std::vector<int64_t>{0},   // begin mask
        std::vector<int64_t>{0});  // end mask
}
}  // namespace
}  // namespace detail

namespace opset_1 {
ov::OutputVector prior_box(const ov::frontend::onnx::Node& node) {
    auto inputs = node.get_ov_inputs();
    FRONT_END_GENERAL_CHECK(inputs.size() == 2, "Invalid number of inputs");

    auto output_shape = std::make_shared<v3::ShapeOf>(inputs[0]);
    auto image_shape = std::make_shared<v3::ShapeOf>(inputs[1]);
    auto output_shape_slice = detail::make_slice(output_shape, 2, 4);
    auto image_shape_slice = detail::make_slice(image_shape, 2, 4);

    ov::op::v8::PriorBox::Attributes attrs;
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

    auto axes = v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0});

    return {
        std::make_shared<v0::Unsqueeze>(std::make_shared<v8::PriorBox>(output_shape_slice, image_shape_slice, attrs),
                                        axes)};
}

ov::OutputVector prior_box_clustered(const ov::frontend::onnx::Node& node) {
    auto inputs = node.get_ov_inputs();
    FRONT_END_GENERAL_CHECK(inputs.size() == 2, "Invalid number of inputs");

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

    auto output_shape = std::make_shared<v3::ShapeOf>(inputs[0]);
    auto image_shape = std::make_shared<v3::ShapeOf>(inputs[1]);
    auto output_shape_slice = detail::make_slice(output_shape, 2, 4);
    auto image_shape_slice = detail::make_slice(image_shape, 2, 4);

    v0::PriorBoxClustered::Attributes attrs{};
    attrs.widths = node.get_attribute_value<std::vector<float>>("width");
    attrs.heights = node.get_attribute_value<std::vector<float>>("height");
    attrs.clip = static_cast<bool>(node.get_attribute_value<int64_t>("clip", 0));
    attrs.variances = node.get_attribute_value<std::vector<float>>("variance", {0.1f});
    attrs.step_heights = node.get_attribute_value<float>("step_h", 0.0f);
    attrs.step_widths = node.get_attribute_value<float>("step_w", 0.0f);
    attrs.step = node.get_attribute_value<float>("step", 0.0f);
    attrs.offset = node.get_attribute_value<float>("offset", 0.0f);

    auto axes = v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0});

    return {std::make_shared<v0::Unsqueeze>(
        std::make_shared<v0::PriorBoxClustered>(output_shape_slice, image_shape_slice, attrs),
        axes)};
}

static bool register_multiple_translators(void) {
    ONNX_OP_M("PriorBox", OPSET_SINCE(1), org_openvinotoolkit::opset_1::prior_box, OPENVINO_ONNX_DOMAIN);
    ONNX_OP_M("PriorBoxClustered",
              OPSET_SINCE(1),
              org_openvinotoolkit::opset_1::prior_box_clustered,
              OPENVINO_ONNX_DOMAIN);
    return true;
}

static bool registered = register_multiple_translators();
}  // namespace opset_1
}  // namespace org_openvinotoolkit
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
