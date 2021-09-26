// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/prior_box.hpp"

#include <node_context.hpp>

#include "default_opset.hpp"

namespace ngraph {
namespace frontend {
namespace pdpd {
namespace op {
using namespace default_opset;
using namespace element;
namespace detail {
namespace {
std::shared_ptr<StridedSlice> make_slice(std::shared_ptr<ngraph::Node> node, int64_t start, int64_t end) {
    return std::make_shared<StridedSlice>(node,
                                          Constant::create(i64, Shape{1}, std::vector<int64_t>{start}),
                                          Constant::create(i64, Shape{1}, std::vector<int64_t>{end}),
                                          std::vector<int64_t>{0},   // begin mask
                                          std::vector<int64_t>{0});  // end mask
}
}  // namespace
}  // namespace detail
NamedOutputs prior_box(const NodeContext& node) {
    auto input = node.get_ng_input("Input");
    auto Image = node.get_ng_input("Image");
    auto input_shape = std::make_shared<ShapeOf>(input);
    auto Image_shape = std::make_shared<ShapeOf>(Image);
    auto output_shape_slice = detail::make_slice(input_shape, 2, 4);
    auto image_shape_slice = detail::make_slice(Image_shape, 2, 4);

    ngraph::op::PriorBoxAttrs attrs;
    attrs.min_size = node.get_attribute<std::vector<float>>("min_sizes", {});
    attrs.max_size = node.get_attribute<std::vector<float>>("max_sizes", {});
    attrs.aspect_ratio = node.get_attribute<std::vector<float>>("aspect_ratios", {1.0});
    attrs.flip = node.get_attribute<bool>("flip", false);
    attrs.clip = node.get_attribute<bool>("clip", false);
    attrs.step = node.get_attribute<float>("step_w", 0);

    attrs.offset = node.get_attribute<float>("offset", 0.5);
    attrs.variance = node.get_attribute<std::vector<float>>("variances", {0.1, 0.1, 0.2, 0.2});

    bool min_max_aspect_ratios_order = node.get_attribute<bool>("min_max_aspect_ratios_order", false);

    auto ov_prior_box_node = std::make_shared<PriorBox>(output_shape_slice, image_shape_slice, attrs);

    auto split_axis_node = Constant::create(i64, ngraph::Shape{}, {0});
    auto node_prior_box_split = std::make_shared<Split>(ov_prior_box_node, split_axis_node, 2);

    auto node_boxes_origin = node_prior_box_split->output(0);
    auto node_variances_origin = node_prior_box_split->output(1);

    auto out_shape =
        std::make_shared<Concat>(NodeVector{output_shape_slice, Constant::create<int64_t>(i64, {2}, {-1, 4})}, 0);

    auto node_boxes_reshape = std::make_shared<Reshape>(node_boxes_origin, out_shape, true);
    auto node_variances_reshape = std::make_shared<Reshape>(node_variances_origin, out_shape, true);

    int64_t total_aspect_ratios = ngraph::op::PriorBox::normalized_aspect_ratio(attrs.aspect_ratio, attrs.flip).size();
    if ((total_aspect_ratios > 1) && !attrs.min_size.empty() && !attrs.max_size.empty() &&
        !min_max_aspect_ratios_order) {
        std::vector<int64_t> mask{1, 1, 1, 0, 1};
        int64_t min_size_len = static_cast<int64_t>(attrs.min_size.size());

        auto out_shape_div_numpri = std::make_shared<Concat>(
            NodeVector{output_shape_slice, Constant::create<int64_t>(i64, {3}, {min_size_len, -1, 4})},
            0);
        auto node_boxes_div_numpri = std::make_shared<Reshape>(node_boxes_reshape, out_shape_div_numpri, true);

        auto slice_begin_min = Constant::create(i64, Shape{5}, std::vector<int64_t>{0, 0, 0, 0, 0});
        auto slice_end_min = std::make_shared<Concat>(
            NodeVector{output_shape_slice, Constant::create<int64_t>(i64, {3}, {min_size_len, 1, 4})},
            0);
        auto slice_min_node =
            std::make_shared<StridedSlice>(node_boxes_div_numpri, slice_begin_min, slice_end_min, mask, mask);

        auto slice_begin_max = Constant::create(i64, Shape{5}, std::vector<int64_t>{0, 0, 0, 1, 0});
        auto slice_end_max = std::make_shared<Concat>(
            NodeVector{output_shape_slice, Constant::create<int64_t>(i64, {3}, {min_size_len, 2, 4})},
            0);
        auto slice_max_node =
            std::make_shared<StridedSlice>(node_boxes_div_numpri, slice_begin_max, slice_end_max, mask, mask);

        auto slice_begin_aspect_ratios = Constant::create(i64, Shape{5}, std::vector<int64_t>{0, 0, 0, 2, 0});
        auto slice_end_aspect_ratios = std::make_shared<Concat>(
            NodeVector{output_shape_slice,
                       Constant::create<int64_t>(i64, {3}, {min_size_len, 2 + (total_aspect_ratios - 1), 4})},
            0);
        auto slice_aspect_ratios_node = std::make_shared<StridedSlice>(node_boxes_div_numpri,
                                                                       slice_begin_aspect_ratios,
                                                                       slice_end_aspect_ratios,
                                                                       mask,
                                                                       mask);

        auto node_boxes_div_numpri_reorder =
            std::make_shared<Concat>(NodeVector{slice_min_node, slice_aspect_ratios_node, slice_max_node}, 3);
        node_boxes_reshape = std::make_shared<Reshape>(node_boxes_div_numpri_reorder, out_shape, true);
    }

    NamedOutputs outputs;
    outputs["Boxes"] = {node_boxes_reshape};
    outputs["Variances"] = {node_variances_reshape};
    return outputs;
}
}  // namespace op
}  // namespace pdpd
}  // namespace frontend
}  // namespace ngraph
