// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_models/builders.hpp"
#include "shared_test_classes/subgraph/reduce_eltwise.hpp"

namespace SubgraphTestsDefinitions {
std::string ReduceEltwiseTest::getTestCaseName(const testing::TestParamInfo<ReduceEltwiseParamsTuple> &obj) {
    std::vector<size_t> inputShapes;
    std::vector<int> axes;
    ov::test::utils::OpType opType;
    bool keepDims;
    InferenceEngine::Precision netPrecision;
    std::string targetName;
    std::tie(inputShapes, axes, opType, keepDims, netPrecision, targetName) = obj.param;

    std::ostringstream result;
    result << "IS=" << ov::test::utils::vec2str(inputShapes) << "_";
    result << "axes=" << ov::test::utils::vec2str(axes) << "_";
    result << "opType=" << opType << "_";
    if (keepDims) result << "KeepDims_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetName;
    return result.str();
}

void ReduceEltwiseTest::SetUp() {
    std::vector<size_t> inputShape;
    std::vector<int> axes;
    ov::test::utils::OpType opType;
    bool keepDims;
    InferenceEngine::Precision netPrecision;
    std::tie(inputShape, axes, opType, keepDims, netPrecision, targetDevice) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

    std::vector<size_t> shapeAxes;
    switch (opType) {
        case ov::test::utils::OpType::SCALAR: {
            if (axes.size() > 1)
                FAIL() << "In reduce op if op type is scalar, 'axis' input's must contain 1 element";
            break;
        }
        case ov::test::utils::OpType::VECTOR: {
            shapeAxes.push_back(axes.size());
            break;
        }
        default:
            FAIL() << "Reduce op doesn't support operation type: " << opType;
    }
    auto reductionAxesNode = std::dynamic_pointer_cast<ngraph::Node>(
                             std::make_shared<ngraph::opset3::Constant>(ngraph::element::Type_t::i64, ngraph::Shape(shapeAxes), axes));

    auto reduce = std::make_shared<ngraph::opset3::ReduceSum>(paramOuts[0], reductionAxesNode, keepDims);

    std::vector<size_t> constShape(reduce.get()->get_output_partial_shape(0).rank().get_length(), 1);
    ASSERT_GT(constShape.size(), 2);
    constShape[2] = inputShape.back();
    auto constant = ngraph::builder::makeConstant<float>(ngPrc, constShape, {}, true);
    auto eltw = ngraph::builder::makeEltwise(reduce, constant, ngraph::helpers::EltwiseTypes::MULTIPLY);
    ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(eltw)};
    function = std::make_shared<ngraph::Function>(results, params, "ReduceEltwise");
}
} // namespace SubgraphTestsDefinitions
