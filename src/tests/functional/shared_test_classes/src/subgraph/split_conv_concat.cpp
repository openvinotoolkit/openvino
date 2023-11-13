// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/split_conv_concat.hpp"

#include "common_test_utils/data_utils.hpp"
#include "ie_common.h"
#include "ov_models/builders.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

std::string SplitConvConcat::getTestCaseName(const testing::TestParamInfo<ov::test::BasicParams>& obj) {
    ov::element::Type element_type;
    ov::Shape inputShapes;
    std::string targetDevice;
    std::tie(element_type, inputShapes, targetDevice) = obj.param;

    std::ostringstream result;
    result << "IS=" << ov::test::utils::vec2str(inputShapes) << "_";
    result << "ET=" << element_type << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void SplitConvConcat::SetUp() {
    configure_test(this->GetParam());
}

void SplitConvConcatBase::configure_test(const ov::test::BasicParams& param) {
    ov::Shape inputShape;
    ov::element::Type element_type;
    std::tie(element_type, inputShape, targetDevice) = param;

    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(element_type, ov::Shape(inputShape))};

    auto split = ngraph::builder::makeSplit(params[0], element_type, 2, 1);

    std::vector<float> filterWeights1;
    std::vector<float> filterWeights2;
    if (targetDevice == ov::test::utils::DEVICE_GNA) {
        filterWeights1 = ov::test::utils::generate_float_numbers(8 * inputShape[1] / 2 * 3, -0.2f, 0.2f);
        filterWeights2 = ov::test::utils::generate_float_numbers(8 * inputShape[1] / 2 * 3, -0.2f, 0.2f);
    }
    auto conv1 = ngraph::builder::makeConvolution(split->output(0),
                                                  element_type,
                                                  {1, 3},
                                                  {1, 1},
                                                  {0, 0},
                                                  {0, 0},
                                                  {1, 1},
                                                  ov::op::PadType::VALID,
                                                  8,
                                                  false,
                                                  filterWeights1);
    auto relu1 = std::make_shared<ov::op::v0::Relu>(conv1);

    auto conv2 = ngraph::builder::makeConvolution(split->output(1),
                                                  element_type,
                                                  {1, 3},
                                                  {1, 1},
                                                  {0, 0},
                                                  {0, 0},
                                                  {1, 1},
                                                  ov::op::PadType::VALID,
                                                  8,
                                                  false,
                                                  filterWeights2);
    auto relu2 = std::make_shared<ov::op::v0::Relu>(conv2);
    auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{relu1->output(0), relu2->output(0)}, 1);

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(concat)};
    function = std::make_shared<ov::Model>(results, params, "SplitConvConcat");
}

}  // namespace test
}  // namespace ov

namespace SubgraphTestsDefinitions {

std::string SplitConvConcat::getTestCaseName(const testing::TestParamInfo<LayerTestsUtils::basicParams>& obj) {
    InferenceEngine::Precision precision;
    InferenceEngine::SizeVector inputShapes;
    std::string targetDevice;
    std::tie(precision, inputShapes, targetDevice) = obj.param;
    auto element_type = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(precision);

    std::ostringstream result;
    result << "IS=" << ov::test::utils::vec2str(inputShapes) << "_";
    result << "ET=" << element_type << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void SplitConvConcat::SetUp() {
    InferenceEngine::Precision precision;
    InferenceEngine::SizeVector inputShapes;
    std::tie(precision, inputShapes, targetDevice) = this->GetParam();
    auto element_type = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(precision);
    ov::Shape shape = inputShapes;

    ov::test::BasicParams param(element_type, shape, targetDevice);
    configure_test(param);
}

}  // namespace SubgraphTestsDefinitions
