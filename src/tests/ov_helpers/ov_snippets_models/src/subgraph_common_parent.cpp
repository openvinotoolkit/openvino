// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_common_parent.hpp"

#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/split.hpp"
#include "snippets/op/result.hpp"
#include "snippets/op/subgraph.hpp"

namespace ov {
namespace test {
namespace snippets {

std::shared_ptr<ov::Model> CommonParentTokenizationFunction::initOriginal() const {
    auto data0 = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    const auto axis = op::v0::Constant::create(element::i64, Shape{}, {0});
    auto split = std::make_shared<op::v1::Split>(data0, axis, 2);

    auto split_out0 = split->output(0);

    // Two eltwise branches that share the same Split output.
    // Both are tokenizable; they will be merged into a single Subgraph
    // when the downstream Add (Add2) is processed, exercising the parameter
    // deduplication path (shared external_input must be reused, not duplicated).
    const auto const1 =
        std::make_shared<op::v0::Constant>(precision, split_out0.get_shape(), std::vector<float>{0.5f, 1.5f});
    auto add = std::make_shared<op::v1::Add>(split_out0, const1);

    const auto const2 =
        std::make_shared<op::v0::Constant>(precision, split_out0.get_shape(), std::vector<float>{2.0f, 0.25f});
    auto mul = std::make_shared<op::v1::Multiply>(split_out0, const2);

    auto add2 = std::make_shared<op::v1::Add>(mul, add);

    return std::make_shared<ov::Model>(OutputVector{add2}, ParameterVector{data0});
}

std::shared_ptr<ov::Model> CommonParentTokenizationFunction::initReference() const {
    auto data0 = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    const auto axis = op::v0::Constant::create(element::i64, Shape{}, {0});
    auto split = std::make_shared<op::v1::Split>(data0, axis, 2);

    auto split_out0 = split->output(0);

    const auto const1 =
        std::make_shared<op::v0::Constant>(precision, split_out0.get_shape(), std::vector<float>{0.5f, 1.5f});
    const auto const2 =
        std::make_shared<op::v0::Constant>(precision, split_out0.get_shape(), std::vector<float>{2.0f, 0.25f});

    // After tokenization, Add, Mul and Add2 are merged into one Subgraph.
    // External inputs: split_out0, const2, const1.
    //
    // Merge order (driven by Add2's input_values order: [Mul, Add]):
    //   1. Mul's inputs are processed first: split_out0 -> ext[0], const2 -> ext[1].
    //   2. Add's inputs are processed second: split_out0 is already at ext[0]
    //      (deduplication), const1 -> ext[2].
    //
    // The deduplication sets current_input_index = 0 (split_out0 position in
    // external_inputs) and replaces Add's body parameter with Mul's.
    auto indata0 = std::make_shared<op::v0::Parameter>(precision, split_out0.get_shape());
    auto indata1 = std::make_shared<op::v0::Parameter>(precision, const2->get_output_shape(0));
    auto indata2 = std::make_shared<op::v0::Parameter>(precision, const1->get_output_shape(0));

    auto add = std::make_shared<op::v1::Add>(indata0, indata2);
    auto mul = std::make_shared<op::v1::Multiply>(indata0, indata1);
    auto add2 = std::make_shared<op::v1::Add>(mul, add);

    auto snippets_result = std::make_shared<ov::snippets::op::Result>(add2);
    auto subgraph = std::make_shared<ov::snippets::op::Subgraph>(
        OutputVector{split_out0, const2, const1},
        std::make_shared<ov::Model>(OutputVector{snippets_result}, ParameterVector{indata0, indata1, indata2}));

    return std::make_shared<ov::Model>(OutputVector{subgraph}, ParameterVector{data0});
}

}  // namespace snippets
}  // namespace test
}  // namespace ov
