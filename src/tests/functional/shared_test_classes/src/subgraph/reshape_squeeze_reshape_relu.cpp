// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/reshape_squeeze_reshape_relu.hpp"

#include "ov_models/builders.hpp"

namespace ov {
namespace test {

std::string ReshapeSqueezeReshapeRelu::getTestCaseName(
    const testing::TestParamInfo<ReshapeSqueezeReshapeReluTuple>& obj) {
    ShapeAxesTuple squeezeShape;
    ov::element::Type element_type;
    std::string targetName;
    ov::test::utils::SqueezeOpType opType;
    std::tie(squeezeShape, element_type, targetName, opType) = obj.param;
    std::ostringstream results;
    results << "OpType=" << opType;
    results << "IS=" << ov::test::utils::vec2str(squeezeShape.first) << "_";
    results << "indices=" << ov::test::utils::vec2str(squeezeShape.second) << "_";
    results << "netPRC=" << element_type << "_";
    results << "targetDevice=" << targetName << "_";
    return results.str();
}

void ReshapeSqueezeReshapeRelu::SetUp() {
    ShapeAxesTuple squeezeShape;
    ov::element::Type element_type;
    ov::test::utils::SqueezeOpType opType;
    std::tie(squeezeShape, element_type, targetDevice, opType) = this->GetParam();
    const size_t input_dim = ov::shape_size(squeezeShape.first);
    std::vector<size_t> shape_input{1, input_dim};
    ov::ParameterVector input{std::make_shared<ov::op::v0::Parameter>(element_type, ov::Shape(shape_input))};
    auto reshape1_pattern = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                                   ov::Shape{squeezeShape.first.size()},
                                                                   squeezeShape.first);
    auto reshape1 = std::make_shared<ov::op::v1::Reshape>(input[0], reshape1_pattern, false);
    auto squeeze = ngraph::builder::makeSqueezeUnsqueeze(reshape1, ov::element::i64, squeezeShape.second, opType);
    auto reshape2_pattern =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{2}, std::vector<size_t>{1, input_dim});
    auto reshape2 = std::make_shared<ov::op::v1::Reshape>(squeeze, reshape2_pattern, false);
    auto func = std::make_shared<ov::op::v0::Relu>(reshape2);
    std::string squeezeType;

    function = std::make_shared<ov::Model>(func, input, "reshape_squeeze_reshape_relu");
}

}  // namespace test
}  // namespace ov
