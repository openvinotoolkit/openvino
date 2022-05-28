// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/layer_norm.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>

#include "default_opset.hpp"
#include "exceptions.hpp"
#include "ngraph/axis_set.hpp"
#include "ngraph/builder/norm.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/validation_util.hpp"

namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
//    Decompose LayerNorm(x) to MVN(x) * gamma + beta
OutputVector layer_norm(const Node& node) {
    auto nodes = node.get_ng_inputs();
    auto num_nodes = nodes.size();
    NGRAPH_CHECK(num_nodes == 3,
                 "LayerNormalization takes 3 inputs. Provided " + std::to_string(num_nodes));

    // input
    Output<ngraph::Node> input = nodes[0];
    float eps = node.get_attribute_value<float>("epsilon");
    // reduce over hidden_size
    int hidden_size_dim = 2;
    const auto reduction_axes = default_opset::Constant::create(element::i32, Shape{1}, {hidden_size_dim});
    std::shared_ptr<ngraph::Node> result =
        std::make_shared<default_opset::MVN>(input, reduction_axes, true, eps, ngraph::op::MVNEpsMode::INSIDE_SQRT);
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
