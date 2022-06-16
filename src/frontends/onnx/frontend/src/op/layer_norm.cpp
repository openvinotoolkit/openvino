// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/layer_norm.hpp"
#include "ngraph/node.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>

#include "default_opset.hpp"
#include "exceptions.hpp"
#include "ngraph/validation_util.hpp"

#include "ngraph/axis_set.hpp"
#include "ngraph/op/mvn.hpp"
#include "ngraph/opsets/opset5.hpp"
#include "ngraph/validation_util.hpp"

namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
//    Decompose LayerNorm(x) to MVN(x) * gamma + beta
OutputVector layer_norm(const Node& node) {
    auto nodes = node.get_ng_inputs();
    auto num_nodes = nodes.size();
    NGRAPH_CHECK(num_nodes == 2 || num_nodes == 3,
                 "LayerNormalization takes 2/3 inputs. Provided " + std::to_string(num_nodes));

    // input
    Output<ngraph::Node> input = nodes[0];

    float eps = node.get_attribute_value<float>("epsilon", 1e-5);
    int64_t axis = node.get_attribute_value<int64_t>("axis", 0);
    //const std::int32_t stash_type{node.get_attribute_value<std::int32_t>("stash_type", 1)};
    std::vector<int64_t> reduction_axes;
    const auto input_shape_size = input.get_partial_shape().rank().get_max_length();
    if (axis >= input_shape_size)
        axis = input_shape_size-1;
    for (int64_t id=axis; id<input_shape_size; id++) {
        reduction_axes.push_back(id);
    }
    auto const_axes = default_opset::Constant::create(element::i64, Shape{reduction_axes.size()}, reduction_axes);
    std::shared_ptr<ngraph::Node> result =
        std::make_shared<default_opset::MVN>(input, const_axes, true, eps, ngraph::op::MVNEpsMode::INSIDE_SQRT);
    // multiply by gamma
    result = std::make_shared<default_opset::Multiply>(result, nodes[1]);
    // add beta if available
    if (num_nodes > 2) {
        result = std::make_shared<default_opset::Add>(result, nodes[2]);
    }
    // spec mentions three outputs (output, mean, inv_std_var) while we support only first one, but:
    // - onnxruntime also doesn't support the last two
    // - we'd have to unroll MVN to have them
    return result->outputs();
}
}  // namespace set_1

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph

