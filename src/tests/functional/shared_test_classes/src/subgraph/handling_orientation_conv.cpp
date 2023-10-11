// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/handling_orientation_conv.hpp"

namespace SubgraphTestsDefinitions {
    std::string HandlingOrientationClass::getTestCaseName(const testing::TestParamInfo<HandlingOrientationParams> &obj) {
        InferenceEngine::Precision netPrecision;
        std::string targetName;
        std::map<std::string, std::string> configuration;
        std::tie(netPrecision, targetName, configuration) = obj.param;
        std::ostringstream results;

        results << "netPRC=" << netPrecision.name() << "_";
        results << "targetDevice=" << targetName << "_";
        return results.str();
    }

    void HandlingOrientationClass::SetUp() {
        InferenceEngine::Precision netPrecision;
        std::tie(netPrecision, targetDevice, configuration) = this->GetParam();
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape{1, 336}),
                                   std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape{1, 336})};
        std::vector<size_t> outFormShapes1 = { 1, 1, 168, 2 };
        std::vector<size_t> outFormShapes2 = { 1, 336, 1, 1 };
        auto pattern1 = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{ 4 }, outFormShapes1);
        auto reshape1 = std::make_shared<ngraph::opset1::Reshape>(params[0], pattern1, false);

        auto pattern2 = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{ 4 }, outFormShapes2);
        auto reshape2 = std::make_shared<ngraph::opset1::Reshape>(params[1], pattern2, false);

        auto permute1 = std::make_shared<ngraph::opset1::Transpose>(reshape1,
                                                                    ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{ 4 }, { 0, 3, 1, 2 }));

        auto conv1 = ngraph::builder::makeConvolution(permute1, ngPrc, { 1, 8 }, { 1, 1 }, { 0, 0 }, { 0, 0 }, { 1, 1 },
                                                      ngraph::op::PadType::VALID, 12);

        auto permute2 = std::make_shared<ngraph::opset1::Transpose>(conv1,
                                                                    ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{ 4 }, { 0, 2, 3, 1 }));

        auto conv2 = ngraph::builder::makeConvolution(reshape2, ngPrc, { 1, 1 }, { 1, 1 }, { 0, 0 }, { 0, 0 }, { 1, 1 },
                                                      ngraph::op::PadType::VALID, 336);

        std::vector<size_t> outFormShapes3 = { 1, 1932 };
        std::vector<size_t> outFormShapes4 = { 1, 336 };
        auto pattern3 = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{ 2 }, outFormShapes3);
        auto pattern4 = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{ 2 }, outFormShapes4);
        auto reshape3 = std::make_shared<ngraph::opset1::Reshape>(permute2, pattern3, false);
        auto reshape4 = std::make_shared<ngraph::opset1::Reshape>(conv2, pattern4, false);
        ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(reshape3),
                                      std::make_shared<ngraph::opset1::Result>(reshape4)};
        function = std::make_shared<ngraph::Function>(results, params, "RemovePermutationPass");
    }
} // namespace SubgraphTestsDefinitions
