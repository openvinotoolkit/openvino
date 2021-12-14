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
            ngraph::helpers::castOps2Nodes<ov::op::v0::Parameter>(params));

    auto p1 = std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::PartialShape(inputShapes[0]));
    auto p2 = std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::PartialShape(inputShapes[1]));
    auto p3 = std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::PartialShape(inputShapes[0]));

    auto thenOp = std::make_shared<ov::op::v1::Add>(p1, p2);
    auto res1 = std::make_shared<ov::op::v0::Result>(thenOp);
    auto res2 = std::make_shared<ov::op::v0::Result>(p3);

    auto thenBody = std::make_shared<ov::Model>(ov::OutputVector{res1}, ov::ParameterVector{p1, p2});
    auto elseBody = std::make_shared<ov::Model>(ov::OutputVector{res2}, ov::ParameterVector{p3});

    auto condOp = ngraph::builder::makeConstant<bool>(ov::element::Type_t::boolean, {1}, {condition});
    auto ifOp = std::make_shared<ov::op::v8::If>(condOp);
    ifOp->set_then_body(thenBody);
    ifOp->set_else_body(elseBody);
    ifOp->set_input(paramOuts[0], p1, p3);
    ifOp->set_input(paramOuts[1], p2, nullptr);
    auto res = ifOp->set_output(res1, res2);

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(res)};
    function = std::make_shared<ov::Model>(results, params, "simpleIf");
}

void SimpleIf2OutTest::SetUp() {
    std::vector<std::vector<size_t>> inputShapes;
    auto netPrecision = InferenceEngine::Precision::UNSPECIFIED;
    bool condition;
    std::tie(inputShapes, netPrecision, condition, targetDevice) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShapes[0], inputShapes[1]});
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ov::op::v0::Parameter>(params));

    auto p1 = std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::PartialShape(inputShapes[0]));
    auto p2 = std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::PartialShape(inputShapes[1]));
    auto p3 = std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::PartialShape(inputShapes[0]));
    auto p4 = std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::PartialShape(inputShapes[1]));

    auto thenOp = std::make_shared<ov::op::v1::Add>(p1, p2);
    auto res1 = std::make_shared<ov::op::v0::Result>(thenOp);
    auto res2 = std::make_shared<ov::op::v0::Result>(thenOp);
    auto res3 = std::make_shared<ov::op::v0::Result>(p3);
    auto res4 = std::make_shared<ov::op::v0::Result>(p4);

    auto thenBody = std::make_shared<ov::Model>(ov::OutputVector{res1, res2}, ov::ParameterVector{p1, p2});
    auto elseBody = std::make_shared<ov::Model>(ov::OutputVector{res3, res4}, ov::ParameterVector{p3, p4});

    auto condOp = ngraph::builder::makeConstant<bool>(ov::element::Type_t::boolean, {1}, {condition});
    auto ifOp = std::make_shared<ov::op::v8::If>(condOp);
    ifOp->set_then_body(thenBody);
    ifOp->set_else_body(elseBody);
    ifOp->set_input(paramOuts[0], p1, p3);
    ifOp->set_input(paramOuts[1], p2, p4);
    auto ifRes1 = ifOp->set_output(res1, res3);
    auto ifRes2 = ifOp->set_output(res2, res4);

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(ifRes1), std::make_shared<ov::op::v0::Result>(ifRes2)};
    function = std::make_shared<ov::Model>(results, params, "simpleIf2Out");
}

void SimpleIfNotConstConditionTest::SetUp() {
    std::vector<std::vector<size_t>> inputShapes;
    auto netPrecision = InferenceEngine::Precision::UNSPECIFIED;
    std::tie(inputShapes, netPrecision, condition, targetDevice) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    ov::ParameterVector params {
        ngraph::builder::makeParams(ngPrc, {inputShapes[0]})[0],
        ngraph::builder::makeParams(ngPrc, {inputShapes[1]})[0],
        ngraph::builder::makeParams(ov::element::boolean, { {"condition", {1}} })[0]
    };
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ov::op::v0::Parameter>(params));

    auto p1 = std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::PartialShape(inputShapes[0]));
    auto p2 = std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::PartialShape(inputShapes[1]));
    auto p3 = std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::PartialShape(inputShapes[0]));
    auto p4 = std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::PartialShape(inputShapes[1]));

    auto thenOp = std::make_shared<ov::op::v1::Add>(p1, p2);
    auto res1 = std::make_shared<ov::op::v0::Result>(thenOp);
    auto res2 = std::make_shared<ov::op::v0::Result>(thenOp);
    auto res3 = std::make_shared<ov::op::v0::Result>(p3);
    auto res4 = std::make_shared<ov::op::v0::Result>(p4);

    auto thenBody = std::make_shared<ov::Model>(ov::OutputVector{res1, res2}, ov::ParameterVector{p1, p2});
    auto elseBody = std::make_shared<ov::Model>(ov::OutputVector{res3, res4}, ov::ParameterVector{p3, p4});

    auto ifOp = std::make_shared<ov::op::v8::If>(paramOuts[2]);
    ifOp->set_then_body(thenBody);
    ifOp->set_else_body(elseBody);
    ifOp->set_input(paramOuts[0], p1, p3);
    ifOp->set_input(paramOuts[1], p2, p4);
    auto ifRes1 = ifOp->set_output(res1, res3);
    auto ifRes2 = ifOp->set_output(res2, res4);

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(ifRes1), std::make_shared<ov::op::v0::Result>(ifRes2)};
    function = std::make_shared<ov::Model>(results, params, "SimpleIfNotConstConditionTest");
}

InferenceEngine::Blob::Ptr SimpleIfNotConstConditionTest::GenerateInput(const InferenceEngine::InputInfo& info) const {
    if (info.name() == "condition") {
        bool conditionArr[1] = { condition };
        return FuncTestUtils::createAndFillBlobWithFloatArray(info.getTensorDesc(), conditionArr, 1);
    } else {
        return FuncTestUtils::createAndFillBlob(info.getTensorDesc());
    }
}

} // namespace SubgraphTestsDefinitions
