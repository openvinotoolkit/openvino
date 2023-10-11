// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/fq_with_mixed_levels.hpp"
#include "ov_models/builders.hpp"

namespace SubgraphTestsDefinitions {

std::string FqWithMixedLevelsTest::getTestCaseName(const testing::TestParamInfo<FqWithMixedLevelsParams>& obj) {
    InferenceEngine::Precision netPrecision;
    std::string targetDevice;
    std::map<std::string, std::string> configuration;
    std::tie(netPrecision, targetDevice, configuration) = obj.param;

    std::ostringstream result;
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice;
    for (auto const& configItem : configuration) {
        result << "_configItem=" << configItem.first << "_" << configItem.second;
    }
    return result.str();
}

void FqWithMixedLevelsTest::SetUp() {
    InferenceEngine::Precision netPrecision;
    std::map<std::string, std::string> tempConfig;
    std::tie(netPrecision, targetDevice, tempConfig) = this->GetParam();
    configuration.insert(tempConfig.begin(), tempConfig.end());

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto unit = [=](const std::shared_ptr<ngraph::Node>& input,
            const std::vector<std::vector<size_t>>& shapes,
            float weights_min, float weights_max,
            size_t level1, const std::vector<std::vector<float>>& data1,
            size_t level2, const std::vector<std::vector<float>>& data2,
            size_t level3, const std::vector<std::vector<float>>& data3) {
        auto sigmoid = std::make_shared<ngraph::opset7::Sigmoid>(input);
        auto fake1 = ngraph::builder::makeFakeQuantize(sigmoid, ngPrc, level1, { 1 }, data1[0], data1[1], data1[2], data1[3]);
        std::vector<float> weights = ov::test::utils::generate_float_numbers(shapes[1][0] * shapes[1][1], weights_min, weights_max);
        auto constant = std::make_shared<ngraph::opset7::Constant>(ngPrc, ngraph::Shape{shapes[1][0], shapes[1][1]}, weights);
        auto fake2 = ngraph::builder::makeFakeQuantize(constant, ngPrc, level2, { 1 }, data2[0], data2[1], data2[2], data2[3]);
        auto matmul = ngraph::builder::makeMatMul(fake1, fake2, false, true);
        auto bias = ngraph::builder::makeConstant(ngPrc, std::vector<size_t>{shapes[0][0], shapes[1][0]}, std::vector<float>{ 1.0 });
        auto add = ngraph::builder::makeEltwise(matmul, bias, ngraph::helpers::EltwiseTypes::ADD);
        return ngraph::builder::makeFakeQuantize(add, ngPrc, level3, { 1 }, data3[0], data3[1], data3[2], data3[3]);
    };

    ov::ParameterVector params {std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape{ 1, 8 })};
    auto input = ngraph::builder::makeFakeQuantize(params[0], ngPrc, std::numeric_limits<uint32_t>::max(), { 1 },
        { -10. }, { 10. }, { -10. }, { 10. });
    input = unit(input,
        {{1, 8}, {8, 8}},
        -20., 20.,
        std::numeric_limits<uint16_t>::max(), {{ -1.0 }, { 1.0 }, { -1.0 }, { 1.0 }},
        std::numeric_limits<uint8_t>::max(), {{ -2.5 }, { 2.5 }, { -2.5 }, { 2.5 }},
        std::numeric_limits<uint32_t>::max(), {{ -5. } , { 5. }, { -5. }, { 5. }});
    input = unit(input,
        {{ 1, 8 }, { 8, 8 }},
        -13., 13.,
        std::numeric_limits<uint16_t>::max(), {{ -1.0 }, { 1.0 }, { -1.0 }, { 1.0 }},
        std::numeric_limits<uint16_t>::max(), {{ -2.5 }, { 2.5 }, { -2.5 }, { 2.5 }},
        std::numeric_limits<uint32_t>::max(), {{ -5. } , { 5. }, { -5. }, { 5. }});
    input = unit(input,
        {{1, 8}, {8, 8}},
        -20., 20.,
        std::numeric_limits<uint16_t>::max(), {{ -1.0 }, { 1.0 }, { -1.0 }, { 1.0 }},
        std::numeric_limits<uint8_t>::max(), {{ -2.5 }, { 2.5 }, { -2.5 }, { 2.5 }},
        std::numeric_limits<uint32_t>::max(), {{ -5. } , { 5. }, { -5. }, { 5. }});
    auto result = std::make_shared<ngraph::opset7::Result>(input);
    function = std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, params, "FqWithMixedLevelsTest");
}

}  // namespace SubgraphTestsDefinitions
