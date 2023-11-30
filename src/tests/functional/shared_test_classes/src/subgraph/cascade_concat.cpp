// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/cascade_concat.hpp"

namespace SubgraphTestsDefinitions {

std::string CascadeConcat::getTestCaseName(const testing::TestParamInfo<CascadeConcatTuple> &obj) {
    std::vector<std::vector<size_t>> input1, input2, input3;
    InferenceEngine::Precision netPrecision;
    std::string targetName;
    bool multioutput;
    std::map<std::string, std::string> additional_config;
    std::tie(input1, input2, input3, netPrecision, multioutput, targetName, additional_config) = obj.param;
    std::ostringstream results;

    results << "IS=" << ov::test::utils::vec2str(input1[0]) << "_";
    results << ov::test::utils::vec2str(input2[0]) << "_";
    results << ov::test::utils::vec2str(input3[0]) << "_";
    results << "netPRC=" << netPrecision.name() << "_";
    results << "Multioutput=" << multioutput << "_";
    results << "targetDevice=" << targetName << "_";
    return results.str();
}

void CascadeConcat::SetUp() {
    std::vector<std::vector<size_t>> input1, input2, input3;
    InferenceEngine::Precision netPrecision;
    std::map<std::string, std::string> additional_config;
    bool multioutput;
    std::tie(input1, input2, input3, netPrecision, multioutput, targetDevice, additional_config) = this->GetParam();
    configuration.insert(additional_config.begin(), additional_config.end());
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    ov::ParameterVector input{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(input1[0])),
                              std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(input2[0])),
                              std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(input3[0]))};

    auto relu1 = std::make_shared<ngraph::opset1::Relu>(input[0]);
    auto relu2 = std::make_shared<ngraph::opset1::Relu>(input[1]);
    auto relu3 = std::make_shared<ngraph::opset1::Relu>(input[2]);
    auto concat = std::make_shared<ngraph::opset1::Concat>(ov::OutputVector{relu1->output(0),
                                                                                relu2->output(0)},
                                                                                1);

    auto reshape_constant = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0});
    auto reshape = std::make_shared<ov::op::v0::Squeeze>(concat, reshape_constant);
    auto reshape2_constant = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0});
    auto reshape2 = std::make_shared<ov::op::v0::Unsqueeze>(reshape, reshape2_constant);

    auto concat2 = std::make_shared<ngraph::opset1::Concat>(ov::OutputVector{reshape2->output(0),
                                                                                 relu3->output(0)},
                                                                                 1);
    ngraph::ResultVector results;
    if (multioutput) {
        auto const_mult = ngraph::builder::makeConstant(ngPrc, ngraph::Shape{1, input1[0][1]+input2[0][1]},
                                                  std::vector<float>{1.01f});
        auto mult = std::make_shared<ngraph::op::v1::Multiply>(concat, const_mult);
        results = ngraph::ResultVector{std::make_shared<ngraph::opset1::Result>(concat2),
                                       std::make_shared<ngraph::opset1::Result>(mult)};
    } else {
        results = ngraph::ResultVector{std::make_shared<ngraph::opset1::Result>(concat2)};
    }
    function = std::make_shared<ngraph::Function>(results, input, "concat_reshape_reshape_concat_mul");
}

std::string CascadeConcatWithMultiConnReshape::getTestCaseName(const testing::TestParamInfo<CascadeConcatWithMultiConnReshapeTuple> &obj) {
    std::vector<size_t> inputShape;
    InferenceEngine::Precision netPrecision;
    std::string targetName;
    std::map<std::string, std::string> additional_config;
    std::tie(inputShape, netPrecision, targetName, additional_config) = obj.param;
    std::ostringstream results;

    results << "IS=" << ov::test::utils::vec2str(inputShape) << "_";
    results << "netPRC=" << netPrecision.name() << "_";
    results << "targetDevice=" << targetName << "_";
    for (auto const& configItem : additional_config) {
        results << "_configItem=" << configItem.first << "_" << configItem.second;
    }
    return results.str();
}

/**
 * Tests a case when 2 concats have Squeeze between them and Concat2 is the second connection of Squeeze output
 * Input     Const1
 *   |         |
 *  Relu       |
 *    |        |
 *      Concat1
 *        |
 *      Squeeze   Const2
 *    |        |   |
 *   Relu1    Concat2
 *    |          |
 * Unsqueeze1   Relu2
 *               |
 *            Unsqueeze2
 */
void CascadeConcatWithMultiConnReshape::SetUp() {
    std::vector<size_t> inputShape;
    InferenceEngine::Precision netPrecision;
    std::map<std::string, std::string> additional_config;
    std::tie(inputShape, netPrecision, targetDevice, additional_config) = this->GetParam();
    configuration.insert(additional_config.begin(), additional_config.end());
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    auto inputShapeSqueezed = inputShape;
    inputShapeSqueezed.insert(std::begin(inputShapeSqueezed), 1);
    ov::ParameterVector input {std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShapeSqueezed))};
    auto relu = std::make_shared<ngraph::opset8::Relu>(input[0]);
    auto const1 = ngraph::builder::makeConstant(ngPrc, inputShapeSqueezed, std::vector<float>{}, true);
    auto concat1 = std::make_shared<ov::op::v0::Concat>(ov::NodeVector{relu, const1}, inputShapeSqueezed.size() - 1);

    auto squeeze_constant = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0});
    auto squeeze = std::make_shared<ov::op::v0::Squeeze>(concat1, squeeze_constant);

    auto relu1 = std::make_shared<ngraph::opset8::Relu>(squeeze);

    auto unsqueeze1_constant = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0});
    auto unsqueeze1 = std::make_shared<ov::op::v0::Unsqueeze>(relu1, unsqueeze1_constant);

    auto const2 = ngraph::builder::makeConstant(ngPrc, inputShape, std::vector<float>{}, true);
    auto concat2 = std::make_shared<ov::op::v0::Concat>(ov::NodeVector{squeeze, const2}, 1);
    // Change concat name to make it the second connection in the map of squeeze output connections
    concat2->set_friendly_name("XConcat");

    auto relu2 = std::make_shared<ngraph::opset8::Relu>(concat2);

    auto unsqueeze2_constant = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0});
    auto unsqueeze2 = std::make_shared<ov::op::v0::Unsqueeze>(relu2, unsqueeze2_constant);

    ngraph::ResultVector results = {std::make_shared<ngraph::opset1::Result>(unsqueeze1),
                                    std::make_shared<ngraph::opset1::Result>(unsqueeze2)};

    function = std::make_shared<ngraph::Function>(results, input, "CascadeConcatWithMultiConnReshapeTest");
}
} // namespace SubgraphTestsDefinitions
