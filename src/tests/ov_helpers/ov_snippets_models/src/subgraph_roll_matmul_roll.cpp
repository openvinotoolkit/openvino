// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_roll_matmul_roll.hpp"
#include <common_test_utils/data_utils.hpp>
#include "openvino/opsets/opset1_decl.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/roll.hpp"

namespace ov {
namespace test {
namespace snippets {

SubgraphRollMatMulRollFunction::SubgraphRollMatMulRollFunction(
    const std::vector<ov::PartialShape>& input_shapes,
    const element::Type input_type) :
    SnippetsFunctionBase(input_shapes, input_type) {
}

std::shared_ptr<ov::Model> SubgraphRollMatMulRollFunction::initOriginal() const {
    const auto parameter1 = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    parameter1->set_friendly_name("parameter1");

    const auto shift = std::make_shared<op::v0::Constant>(ov::element::i32, ov::Shape{ 1 }, std::vector<float>{1});
    const auto axes = std::make_shared<op::v0::Constant>(ov::element::i32, ov::Shape{ 1 }, std::vector<float>{0});

    std::shared_ptr<Node> parent1 = std::make_shared<ov::op::v7::Roll>(parameter1, shift, axes);
    parent1->get_rt_info()["enforceBF16evenForGraphTail"] = true;
    parent1->set_friendly_name("roll1");

    std::shared_ptr<ov::opset1::Parameter> parameter2;
    std::shared_ptr<Node> parent2;

    parameter2 = std::make_shared<ov::opset1::Parameter>(precision, input_shapes[1]);
    parameter2->set_friendly_name("parameter2");

    parent2 = std::make_shared<ov::op::v7::Roll>(parameter2, shift, axes);
    parent2->get_rt_info()["enforceBF16evenForGraphTail"] = true;
    parent2->set_friendly_name("roll2");

    parent1 = std::make_shared<op::v0::MatMul>(parent1, parent2);
    parent1->set_friendly_name("matmul");
    parent1->get_rt_info()["enforceBF16evenForGraphTail"] = true;


    auto roll3 = std::make_shared<ov::op::v7::Roll>(parent1, shift, axes);
    roll3->set_friendly_name("roll3");

    const auto result = std::make_shared<ov::opset1::Result>(roll3);
    result->set_friendly_name("result");

    return std::make_shared<ov::Model>(
        ov::ResultVector{ result },
        parameter2 == nullptr ? ParameterVector{ parameter1 } : ParameterVector{ parameter1, parameter2 },
        "SubgraphTransposeMatMulFunction");
}

}  // namespace snippets
}  // namespace test
}  // namespace ov
