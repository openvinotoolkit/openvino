// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <ie_plugin_config.hpp>
#include "shared_test_classes/single_layer/multinomial.hpp"
#include "ov_models/builders.hpp"
#include "ov_models/utils/ov_helpers.hpp"

namespace ov {
namespace test {
namespace subgraph {

std::string MultinomialTest::getTestCaseName(const testing::TestParamInfo<MultinomialTestParams> &obj) {
    ElementType netType, inType, outType;
    InputShape shape;
    std::int64_t numSamples;
    element::Type_t outputType;
    bool withReplacement;
    bool logProbs;
    TargetDevice targetDevice;
    Config config;
    std::tie(netType, inType, outType, shape, numSamples, outputType, withReplacement, logProbs, targetDevice, config) = obj.param;

    std::ostringstream result;
    result << "NetType=" << netType << "_";
    result << "InType=" << inType << "_";
    result << "OutType=" << outType << "_";
    result << "IS=" << ov::test::utils::partialShape2str({shape.first}) << "_";
    result << "TS=";
    for (const auto& item : shape.second) {
        result << ov::test::utils::vec2str(item) << "_";
    }
    result << "NumSamples=" << numSamples << "_";
    result << "OutputType=" << outputType << "_";
    result << "WithReplacement=" << withReplacement << "_";
    result << "LogProbs=" << logProbs << "_";
    result << "Device=" << targetDevice;

    return result.str();
}

void MultinomialTest::SetUp() {
    InputShape shape;
    ElementType ngPrc;
    std::int64_t numSamples;
    element::Type_t outputType;
    bool withReplacement;
    bool logProbs;

    std::tie(ngPrc, inType, outType, shape, numSamples, outputType, withReplacement, logProbs, targetDevice, configuration) = this->GetParam();
    init_input_shapes({shape});
    ov::ParameterVector params;
    for (auto&& shape : inputDynamicShapes) {
        params.push_back(std::make_shared<ov::op::v0::Parameter>(ngPrc, shape));
    }

    auto numSamplesConstant = std::make_shared<ngraph::opset3::Constant>(
        ngraph::element::Type_t::i64, ov::Shape{1}, numSamples);
    const auto paramOuts =
        ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

    const auto multinomial = std::make_shared<ov::op::v13::Multinomial>(
        paramOuts.at(0),
        numSamplesConstant,
        outputType,
        withReplacement,
        logProbs,
        0,
        2);

    function = std::make_shared<ngraph::Function>(multinomial, params, "Multinomial");
}

} // namespace subgraph
} // namespace test
} // namespace ov
