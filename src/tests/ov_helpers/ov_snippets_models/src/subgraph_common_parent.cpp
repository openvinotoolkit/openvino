// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_common_parent.hpp"

#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/variadic_split.hpp"
#include "snippets/op/result.hpp"
#include "snippets/op/subgraph.hpp"

namespace ov {
namespace test {
namespace snippets {

std::shared_ptr<ov::Model> CommonParentTokenizationFunction::initOriginal() const {
    auto data0 = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    const auto axis = op::v0::Constant::create(element::i64, Shape{}, {0});
    const auto split_lengths = op::v0::Constant::create(element::i64, Shape{2}, {1, 2});
    auto split = std::make_shared<op::v1::VariadicSplit>(data0, axis, split_lengths);

    auto split_out0 = split->output(0);
    auto split_out1 = split->output(1);

    // The first branch orders the shared inputs as [split_out1, split_out0].
    // The second branch uses [split_out0, split_out1], so merging it must
    // reuse body parameters at external input indices 1 and 0 respectively.
    auto add = std::make_shared<op::v1::Add>(split_out1, split_out0);
    auto mul = std::make_shared<op::v1::Multiply>(split_out0, split_out1);
    auto add2 = std::make_shared<op::v1::Add>(add, mul);

    return std::make_shared<ov::Model>(OutputVector{add2}, ParameterVector{data0});
}

std::shared_ptr<ov::Model> CommonParentTokenizationFunction::initReference() const {
    auto data0 = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    const auto axis = op::v0::Constant::create(element::i64, Shape{}, {0});
    const auto split_lengths = op::v0::Constant::create(element::i64, Shape{2}, {1, 2});
    auto split = std::make_shared<op::v1::VariadicSplit>(data0, axis, split_lengths);

    auto split_out0 = split->output(0);
    auto split_out1 = split->output(1);

    // Tokenization orders the merged Subgraph external inputs as
    // [split_out1, split_out0] from the first Add branch.
    auto indata0 = std::make_shared<op::v0::Parameter>(precision, split_out1.get_shape());
    auto indata1 = std::make_shared<op::v0::Parameter>(precision, split_out0.get_shape());

    auto add = std::make_shared<op::v1::Add>(indata0, indata1);
    auto mul = std::make_shared<op::v1::Multiply>(indata1, indata0);
    auto add2 = std::make_shared<op::v1::Add>(add, mul);

    auto snippets_result = std::make_shared<ov::snippets::op::Result>(add2);
    auto subgraph = std::make_shared<ov::snippets::op::Subgraph>(
        OutputVector{split_out1, split_out0},
        std::make_shared<ov::Model>(OutputVector{snippets_result}, ParameterVector{indata0, indata1}));

    return std::make_shared<ov::Model>(OutputVector{subgraph}, ParameterVector{data0});
}

}  // namespace snippets
}  // namespace test
}  // namespace ov
