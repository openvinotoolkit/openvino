// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/builder/split.hpp"

#include "ngraph/opsets/opset1.hpp"

using namespace ngraph;

OutputVector builder::opset1::split(const Output<Node>& value,
                                    const std::vector<int64_t>& split_lengths,
                                    int64_t axis) {
    const auto axis_node = ngraph::opset1::Constant::create(element::i64, Shape{}, {axis});
    const auto split_lengths_node =
        ngraph::opset1::Constant::create(element::i64, Shape{split_lengths.size()}, split_lengths);
    const auto variadic_split = std::make_shared<ngraph::opset1::VariadicSplit>(value, axis_node, split_lengths_node);

    return variadic_split->outputs();
}

OutputVector builder::opset1::split(const Output<Node>& value, int64_t num_splits, int64_t axis) {
    const auto axis_node = ngraph::opset1::Constant::create(element::i64, Shape{}, {axis});
    const auto split = std::make_shared<ngraph::opset1::Split>(value, axis_node, num_splits);

    return split->outputs();
}
