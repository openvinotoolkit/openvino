// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/layer_normalization.hpp"

#include "default_opset.hpp"
#include "exceptions.hpp"
#include "ngraph/axis_set.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/mvn.hpp"
#include "ngraph/validation_util.hpp"
#include "utils/common.hpp"

namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
std::shared_ptr<ngraph::Node> get_dynamic_all_axes_range(const Node& node) {
    const auto input = node.get_ng_inputs().at(0);
    const auto shape_of_input = std::make_shared<default_opset::ShapeOf>(input);
    const auto scalar = default_opset::Constant::create(element::i32, Shape{1}, {0});
    const auto rank_of_input = std::make_shared<default_opset::ShapeOf>(shape_of_input);
    const auto rank_of_input_scalar = std::make_shared<default_opset::Squeeze>(rank_of_input, scalar);
    const auto start = default_opset::Constant::create(element::i32, Shape{}, {0});
    const auto step = default_opset::Constant::create(element::i32, Shape{}, {1});
    return std::make_shared<default_opset::Range>(start, rank_of_input_scalar, step, element::i64);
}

std::shared_ptr<ngraph::Node> get_reduction_axes_from_input(const Node& node) {
    const std::int64_t noop_with_empty_axes = node.get_attribute_value<std::int64_t>("noop_with_empty_axes", 0);
    const auto input = node.get_ng_inputs().at(0);
    if (node.get_ng_inputs().size() > 1) {
        const auto reduction_axes = node.get_ng_inputs().at(1);
        const auto reduction_axes_rank = reduction_axes.get_partial_shape().rank();
        NGRAPH_CHECK(reduction_axes.get_partial_shape().is_static(),
                     "The axes tensor's shape needs to be known(static). Node: ",
                     node.get_description());

        if (reduction_axes_rank.get_length() != 0 && reduction_axes.get_shape() != Shape{0}) {
            return reduction_axes.get_node_shared_ptr();
        }
    }

    if (noop_with_empty_axes) {
        return nullptr;
    } else {
        return get_dynamic_all_axes_range(node);
    }
}

std::shared_ptr<ngraph::Node> get_reduction_axes_from_attr(const Node& node) {
    auto reduction_axes = node.get_attribute_value<std::vector<std::int64_t>>("axis", {});

    const auto input_rank = node.get_ng_inputs().at(0).get_partial_shape().rank();

    if (reduction_axes.empty()) {
        if (input_rank.is_static()) {
            reduction_axes = onnx_import::common::get_monotonic_range<int64_t>(input_rank.get_length());
        } else {
            return get_dynamic_all_axes_range(node);
        }
    }

    if (input_rank.is_static()) {
        CHECK_VALID_NODE(node,
                         static_cast<int64_t>(reduction_axes.size()) <= input_rank.get_length(),
                         "Number of reduction axes (",
                         reduction_axes.size(),
                         ") is larger than the input tensor's rank (",
                         input_rank.get_length(),
                         ")");
    }

    return default_opset::Constant::create(element::i64, Shape{reduction_axes.size()}, reduction_axes);
}

//    Decompose LayerNorm(x) to MVN(x) * gamma + beta
OutputVector layer_normalization(const Node& node) {
    auto nodes = node.get_ng_inputs();
    auto num_nodes = nodes.size();
    NGRAPH_CHECK(num_nodes == 2 || num_nodes == 3,
                 "LayerNormalization takes 2/3 inputs. Provided " + std::to_string(num_nodes));

    // input
    Output<ngraph::Node> input = nodes[0];

    float eps = node.get_attribute_value<float>("epsilon", 1e-5);
    const auto reduction_axes = get_reduction_axes_from_attr(node);
    std::shared_ptr<ngraph::Node> result =
        std::make_shared<default_opset::MVN>(input, reduction_axes, true, eps, ngraph::op::MVNEpsMode::INSIDE_SQRT);
    // multiply by gamma
    result = std::make_shared<default_opset::Multiply>(result, nodes[1]);
    // add beta if available
    if (num_nodes > 2) {
        result = std::make_shared<default_opset::Add>(result, nodes[2]);
    }
    // - we'd have to unroll MVN to have them
    return result->outputs();
}
}  // namespace set_1

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
