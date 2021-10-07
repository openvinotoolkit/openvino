// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <memory>
#include <vector>

#include "default_opset.hpp"
#include "exceptions.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/util/op_types.hpp"
#include "ngraph/validation_util.hpp"
#include "onnx_import/core/null_node.hpp"
#include "op/gather.hpp"
#include "utils/common.hpp"

#include "openvino/opsets/opset8.hpp"

namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_10 {
OutputVector slice(const Node& node) {
    using ngraph::op::is_null;

    OutputVector inputs{node.get_ng_inputs()};
    const auto data = inputs.at(0);
    const auto data_rank = data.get_partial_shape().rank();

    auto starts = inputs.at(1);
    auto ends = inputs.at(2);

    // Slice is calculated over all axes as default
    std::shared_ptr<default_opset::Constant> axes_const;
    if (inputs.size() >= 4 && !is_null(inputs.at(3)))  // axes input provided
    {
        axes_const = ngraph::get_constant_from_source(inputs.at(3));
        CHECK_VALID_NODE(node, axes_const != nullptr, "Axes input must be constant");
    } else {
        CHECK_VALID_NODE(node, data_rank.is_static(), "Data rank must be static when axes input is not provided");
        const size_t data_rank_value = data_rank.get_length();
        axes_const = default_opset::Constant::create(element::i64,
                                                     {data_rank_value},
                                                     common::get_monotonic_range<int64_t>(data_rank_value));
    }
    auto raw_axes_vec = axes_const->cast_vector<int64_t>();
    std::vector<uint64_t> axes_vec = get_normalized_axes_vector(node, data_rank, raw_axes_vec);

    const size_t slice_indices_length = *std::max_element(std::begin(axes_vec), std::end(axes_vec)) + 1;
    const auto begin_end_mask = axes_to_mask(axes_vec, slice_indices_length);

    Output<ngraph::Node> steps;
    if (inputs.size() == 5 && !is_null(inputs.at(4)))  // steps input provided
    {
        steps = inputs.at(4);
    } else {
        steps = default_opset::Constant::create(element::i64,
                                                {slice_indices_length},
                                                std::vector<int64_t>(slice_indices_length, 1));
    }

    starts = adjust_indices_if_needed(starts, axes_vec, slice_indices_length, 0);
    ends = adjust_indices_if_needed(ends, axes_vec, slice_indices_length, 0);
    steps = adjust_indices_if_needed(steps, axes_vec, slice_indices_length, 1);

    return {std::make_shared<default_opset::StridedSlice>(data, starts, ends, steps, begin_end_mask, begin_end_mask)};
}
}  // namespace set_10

namespace set_1 {
OutputVector slice(const Node& node) {
    Output<ngraph::Node> data = node.get_ng_inputs().at(0);
    const auto starts_atr = node.get_attribute_value<std::vector<int64_t>>("starts");
    const auto ends_atr = node.get_attribute_value<std::vector<int64_t>>("ends");

    const auto& starts =
        std::make_shared<default_opset::Constant>(element::i64, Shape{starts_atr.size()}, starts_atr);
    const auto& ends =
        std::make_shared<default_opset::Constant>(element::i64, Shape{ends_atr.size()}, ends_atr);

    auto axes_atr = node.get_attribute_value<std::vector<int64_t>>("axes", std::vector<int64_t>());

    const auto& steps = default_opset::Constant::create(element::i64,
                                        Shape{starts_atr.size()},
                                        std::vector<int64_t>(starts_atr.size(), 1));

    if (axes_atr.empty()) {
        return {std::make_shared<opset8::Slice>(data, starts, ends, steps)};
    } else {
        const auto& axes = std::make_shared<default_opset::Constant>(element::i64, Shape{axes_atr.size()}, axes_atr);
        return {std::make_shared<opset8::Slice>(data, starts, ends, steps, axes)};
    }
}
}  // namespace set_1
}  // namespace op
}  // namespace onnx_import
}  // namespace ngraph
