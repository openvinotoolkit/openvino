// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/output_layers.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>


#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/node_builders/fake_quantize.hpp"


namespace LayerTestsDefinitions {

std::string OutputLayers::getTestCaseName(const testing::TestParamInfo<LayerTestsUtils::LayerTransformationParams>& obj) {
    ov::element::Type netPrecision;
    ov::Shape inputShapes;
    std::string targetDevice;
    ov::pass::low_precision::LayerTransformation::Params params;
    std::tie(netPrecision, inputShapes, targetDevice, params) = obj.param;

    return get_test_case_name_by_params(netPrecision, inputShapes, targetDevice, params);
}


void OutputLayers::SetUp() {
    ov::Shape inputShape;
    ov::element::Type ngPrecision;
    ov::pass::low_precision::LayerTransformation::Params params;
    std::tie(ngPrecision, inputShape, targetDevice, params) = this->GetParam();

    init_input_shapes(ov::PartialShape(inputShape));

    const auto input = std::make_shared<ov::op::v0::Parameter>(ngPrecision, ov::Shape(inputShape));
    input->set_friendly_name("input");

    const float k = 1.f;
    const auto fakeQuantizeOnActivations = ov::test::utils::make_fake_quantize(
        input->output(0), ngPrecision, 256ul, { 1ul },
        { 0.f }, { 255.f / k }, { 0.f }, { 255.f / k });
    fakeQuantizeOnActivations->set_friendly_name("fakeQuantizeOnActivations");

    const auto weights = ov::op::v0::Constant::create(
        ngPrecision,
        ov::Shape{ inputShape[1ul], inputShape[1ul], 1ul, 1ul },
        std::vector<float>(inputShape[1ul] * inputShape[1ul], 1ul));
    weights->set_friendly_name("weights");
    const auto fakeQuantizeOnWeights = ov::test::utils::make_fake_quantize(
        weights, ngPrecision, 256ul, { 1ul },
        { -128.f / k }, { 127.f / k }, { -128.f / k }, { 127.f / k });
    fakeQuantizeOnWeights->set_friendly_name("fakeQuantizeOnWeights");

    std::shared_ptr<ov::op::v1::Convolution> convolution = std::make_shared<ov::op::v1::Convolution>(
        fakeQuantizeOnActivations,
        fakeQuantizeOnWeights,
        ov::Strides{ 1ul, 1ul },
        ov::CoordinateDiff{ 0, 0 },
        ov::CoordinateDiff{ 0, 0 },
        ov::Strides{ 1ul, 1ul });
    convolution->set_friendly_name("convolution");

    ov::ResultVector results {
        std::make_shared<ov::op::v0::Result>(convolution),
        std::make_shared<ov::op::v0::Result>(fakeQuantizeOnActivations)
    };

    function = std::make_shared<ov::Model>(results, ov::ParameterVector { input }, "OutputLayersHandling");
}

TEST_P(OutputLayers, CompareWithRefImpl) {
    run();
};

}  // namespace LayerTestsDefinitions
