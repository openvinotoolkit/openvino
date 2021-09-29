// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/simple_if.hpp"
#include "ngraph_functions/builders.hpp"

namespace SubgraphTestsDefinitions {
std::string SimpleIfTest::getTestCaseName(const testing::TestParamInfo<SimpleIfParamsTuple> &obj) {
    std::vector<std::vector<size_t>> inputShapes;
    InferenceEngine::Precision netPrecision;
    bool condition;
    std::string targetName;
    std::tie(inputShapes, netPrecision, condition, targetName) = obj.param;
    std::ostringstream results;

    results << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
    results << "netPRC=" << netPrecision.name() << "_";
    results << "Cond=" << condition << "_";
    results << "targetDevice=" << targetName << "_";
    return results.str();
}

void SimpleIfTest::SetUp() {
    std::vector<std::vector<size_t>> inputShapes;
    auto netPrecision = InferenceEngine::Precision::UNSPECIFIED;
    bool condition;
    std::tie(inputShapes, netPrecision, condition, targetDevice) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShapes[0], inputShapes[1]});
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

    auto p1 = std::make_shared<ngraph::op::Parameter>(ngPrc, ngraph::PartialShape(inputShapes[0]));
    auto p2 = std::make_shared<ngraph::op::Parameter>(ngPrc, ngraph::PartialShape(inputShapes[1]));
    auto p3 = std::make_shared<ngraph::op::Parameter>(ngPrc, ngraph::PartialShape(inputShapes[0]));

    auto thenOp = std::make_shared<ngraph::op::v1::Add>(p1, p2);
    auto res1 = std::make_shared<ngraph::op::Result>(thenOp);
    auto res2 = std::make_shared<ngraph::op::Result>(p3);

    auto thenBody = std::make_shared<ngraph::Function>(ngraph::OutputVector{res1}, ngraph::ParameterVector{p1, p2});
    auto elseBody = std::make_shared<ngraph::Function>(ngraph::OutputVector{res2}, ngraph::ParameterVector{p3});

    auto condOp = ngraph::builder::makeConstant<bool>(ngraph::element::Type_t::boolean, {1}, {condition});
    auto ifOp = std::make_shared<ngraph::op::v8::If>(condOp);
    ifOp->set_then_body(thenBody);
    ifOp->set_else_body(elseBody);
    ifOp->set_input(paramOuts[0], p1, p3);
    ifOp->set_input(paramOuts[1], p2, nullptr);
    auto res = ifOp->set_output(res1, res2);

    ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(res)};
    function = std::make_shared<ngraph::Function>(results, params, "simpleIf");
}

void SimpleIf2OutTest::SetUp() {
    std::vector<std::vector<size_t>> inputShapes;
    auto netPrecision = InferenceEngine::Precision::UNSPECIFIED;
    bool condition;
    std::tie(inputShapes, netPrecision, condition, targetDevice) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShapes[0], inputShapes[1]});
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

    auto p1 = std::make_shared<ngraph::op::Parameter>(ngPrc, ngraph::PartialShape(inputShapes[0]));
    auto p2 = std::make_shared<ngraph::op::Parameter>(ngPrc, ngraph::PartialShape(inputShapes[1]));
    auto p3 = std::make_shared<ngraph::op::Parameter>(ngPrc, ngraph::PartialShape(inputShapes[0]));
    auto p4 = std::make_shared<ngraph::op::Parameter>(ngPrc, ngraph::PartialShape(inputShapes[1]));

    auto thenOp = std::make_shared<ngraph::op::v1::Add>(p1, p2);
    auto res1 = std::make_shared<ngraph::op::Result>(thenOp);
    auto res2 = std::make_shared<ngraph::op::Result>(thenOp);
    auto res3 = std::make_shared<ngraph::op::Result>(p3);
    auto res4 = std::make_shared<ngraph::op::Result>(p4);

    auto thenBody = std::make_shared<ngraph::Function>(ngraph::OutputVector{res1, res2}, ngraph::ParameterVector{p1, p2});
    auto elseBody = std::make_shared<ngraph::Function>(ngraph::OutputVector{res3, res4}, ngraph::ParameterVector{p3, p4});

    auto condOp = ngraph::builder::makeConstant<bool>(ngraph::element::Type_t::boolean, {1}, {condition});
    auto ifOp = std::make_shared<ngraph::op::v8::If>(condOp);
    ifOp->set_then_body(thenBody);
    ifOp->set_else_body(elseBody);
    ifOp->set_input(paramOuts[0], p1, p3);
    ifOp->set_input(paramOuts[1], p2, p4);
    auto ifRes1 = ifOp->set_output(res1, res3);
    auto ifRes2 = ifOp->set_output(res2, res4);

    ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(ifRes1), std::make_shared<ngraph::opset3::Result>(ifRes2)};
    function = std::make_shared<ngraph::Function>(results, params, "simpleIf2Out");
}
} // namespace SubgraphTestsDefinitions
