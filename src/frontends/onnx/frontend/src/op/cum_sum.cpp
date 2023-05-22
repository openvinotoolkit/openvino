// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/cum_sum.hpp"

#include <memory>

#include "default_opset.hpp"

namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector cum_sum(const Node& node) {
    auto inputs = node.get_ng_inputs();
    auto data = inputs.at(0);
    bool exclusive = node.get_attribute_value<std::int64_t>("exclusive", 0);
    bool reverse = node.get_attribute_value<std::int64_t>("reverse", 0);
    Output<ngraph::Node> axis;

    if (inputs.size() > 1) {
        // optional input, 0-D or 1-D tensor
        const auto& axis_shape = inputs.at(1).get_partial_shape();
        NGRAPH_CHECK(axis_shape.rank().get_length() == 0 || axis_shape.rank().get_length() == 1,
                     "axis shape should be 0-D or 1-D.");
        axis =
            axis_shape.rank().get_length() == 0 ? inputs.at(1) : std::make_shared<default_opset::Squeeze>(inputs.at(1));
    } else {
        axis = default_opset::Constant::create(element::i64, Shape{}, {0});  // default
    }
    return OutputVector{std::make_shared<default_opset::CumSum>(data, axis, exclusive, reverse)};
}

}  // namespace set_1

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
