// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_models/builders.hpp"
#include "shared_test_classes/subgraph/multiple_input_fq.hpp"
#include "common_test_utils/node_builders/eltwise.hpp"
#include "common_test_utils/node_builders/fake_quantize.hpp"

namespace SubgraphTestsDefinitions {

std::string MultipleInputTest::getTestCaseName(const testing::TestParamInfo<multipleInputParams> &obj) {
    std::string targetDevice;
    InferenceEngine::Precision netPrecision;
    size_t inputSize;
    std::map<std::string, std::string> config;
    std::tie(targetDevice, netPrecision, inputSize, config) = obj.param;
    std::ostringstream result;
    result << "netPrecision=" << netPrecision.name() << "_";
    result << "IS=" << inputSize << "_";
    result << "targetDevice=" << targetDevice;
    for (auto const& configItem : config) {
        result << "_configItem=" << configItem.first << "_" << configItem.second;
    }
    return result.str();
}

void MultipleInputTest::SetUp() {
    InferenceEngine::Precision netPrecision;
    std::map<std::string, std::string> config;
    size_t inputSize;
    std::tie(targetDevice, netPrecision, inputSize, config) = this->GetParam();
    configuration.insert(config.begin(), config.end());
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    const float minInput = -10.0;
    const float maxInput = 10.0;
    ov::ParameterVector input{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape{1, inputSize}),
                              std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape{1, inputSize}),
                              std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape{1, inputSize})};
    auto fake1 = ov::test::utils::make_fake_quantize(input[0], ngPrc, std::numeric_limits<uint16_t>::max(), { 1 },
        { minInput }, { maxInput }, { minInput }, { maxInput });
    auto add1 = ov::test::utils::make_eltwise(input[0], fake1, ngraph::helpers::EltwiseTypes::ADD);
    auto fake_add1 = ov::test::utils::make_fake_quantize(add1, ngPrc, std::numeric_limits<uint16_t>::max(), { 1 },
        { 2 * minInput }, { 2 * maxInput }, { 2 * minInput }, { 2 * maxInput });

    auto fake2 = ov::test::utils::make_fake_quantize(input[1], ngPrc, std::numeric_limits<uint16_t>::max(), { 1 },
        { minInput }, { maxInput }, { minInput }, { maxInput });
    auto add2 = ov::test::utils::make_eltwise(input[1], fake2, ngraph::helpers::EltwiseTypes::ADD);
    auto fake_add2 = ov::test::utils::make_fake_quantize(add2, ngPrc, std::numeric_limits<uint16_t>::max(), { 1 },
        { 2 * minInput }, { 2 * maxInput }, { 2 * minInput }, { 2 * maxInput });

    auto add3 = ov::test::utils::make_eltwise(fake_add1, fake_add2, ngraph::helpers::EltwiseTypes::ADD);
    auto fake_add3 = ov::test::utils::make_fake_quantize(add3, ngPrc, std::numeric_limits<uint16_t>::max(), { 1 },
        { 4 * minInput }, { 4 * maxInput }, { 4 * minInput }, { 4 * maxInput });

    auto fake3 = ov::test::utils::make_fake_quantize(input[2], ngPrc, std::numeric_limits<uint16_t>::max(), { 1 },
        { minInput }, { maxInput }, { minInput }, { maxInput });
    auto add4 = ov::test::utils::make_eltwise(fake3, fake_add3, ngraph::helpers::EltwiseTypes::ADD);
    auto fake_add4 = ov::test::utils::make_fake_quantize(add4, ngPrc, std::numeric_limits<uint16_t>::max(), { 1 },
        { 5 * minInput }, { 5 * maxInput }, { 5 * minInput }, { 5 * maxInput });

    auto result = std::make_shared<ov::op::v0::Result>(fake_add4);
    function = std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, input, "multiple_input");
}

}  // namespace SubgraphTestsDefinitions

