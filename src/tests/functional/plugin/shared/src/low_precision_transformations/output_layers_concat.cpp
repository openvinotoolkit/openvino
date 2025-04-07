// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/output_layers_concat.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>


#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/node_builders/fake_quantize.hpp"


namespace LayerTestsDefinitions {

std::string OutputLayersConcat::getTestCaseName(const testing::TestParamInfo<LayerTestsUtils::LayerTransformationParams>& obj) {
    ov::element::Type netPrecision;
    ov::Shape inputShapes;
    std::string targetDevice;
    ov::pass::low_precision::LayerTransformation::Params params;
    std::tie(netPrecision, inputShapes, targetDevice, params) = obj.param;

    return get_test_case_name_by_params(netPrecision, inputShapes, targetDevice, params);
}


/*
*           FQ1     FQ2
*            \      / \
*             \    /   Output
*             Concat
*            /      \
*           /        \
*  Convolution      Output
*        /
*       /
*   Output
*/

void OutputLayersConcat::SetUp() {
    ov::Shape inputShape1;
    ov::element::Type ngPrecision;
    ov::pass::low_precision::LayerTransformation::Params params;
    std::tie(ngPrecision, inputShape1, targetDevice, params) = this->GetParam();

    init_input_shapes({
                              ov::PartialShape(inputShape1),
                              ov::PartialShape(std::vector<ov::Dimension::value_type>({
                                                                                              static_cast<ov::Dimension::value_type>(inputShape1[0]),
                                                                                              static_cast<ov::Dimension::value_type>(inputShape1[1] * 2ul),
                                                                                              static_cast<ov::Dimension::value_type>(inputShape1[2]),
                                                                                              static_cast<ov::Dimension::value_type>(inputShape1[3])
                                                                                      }))
                      });

    const auto input1 = std::make_shared<ov::op::v0::Parameter>(ngPrecision, ov::Shape(inputShape1));
    input1->set_friendly_name("input1");

    const auto fakeQuantize1 = ov::test::utils::make_fake_quantize(
        input1->output(0), ngPrecision, 256ul, { 1ul },
        { 0.f }, { 255.f }, { 0.f }, { 255.f });
    fakeQuantize1->set_friendly_name("fakeQuantize1");

    ASSERT_EQ(4ul, inputShape1.size()) << "unexpected input layout";
    const ov::Shape inputShape2 = { inputShape1[0], inputShape1[1] * 2ul, inputShape1[2], inputShape1[3] };
    const auto input2 = std::make_shared<ov::op::v0::Parameter>(ngPrecision, ov::Shape(inputShape2));
    input2->set_friendly_name("input2");

    const auto fakeQuantize2 = ov::test::utils::make_fake_quantize(
        input2->output(0), ngPrecision, 256ul, { 1ul },
        { 0.f }, { 255.f / 2.f }, { 0.f }, { 255.f / 2.f });
    fakeQuantize2->set_friendly_name("fakeQuantize2");

    const std::shared_ptr<ov::op::v0::Concat> concat = std::make_shared<ov::op::v0::Concat>(
        ov::OutputVector{ fakeQuantize1->output(0), fakeQuantize2->output(0)}, 1);
    concat->set_friendly_name("concat");

    const float k = 1.f;
    const auto weights = ov::op::v0::Constant::create(
        ngPrecision,
        ov::Shape{ inputShape1[1ul] + inputShape2[1ul], inputShape1[1ul] + inputShape2[1ul], 1ul, 1ul },
        std::vector<float>((inputShape1[1ul] + inputShape2[1ul]) * (inputShape1[1ul] + inputShape2[1ul]), 1ul));
    weights->set_friendly_name("weights");
    const auto fakeQuantizeOnWeights = ov::test::utils::make_fake_quantize(
        weights, ngPrecision, 256ul, { 1ul },
        { -128.f / k }, { 127.f / k }, { -128.f / k }, { 127.f / k });
    fakeQuantizeOnWeights->set_friendly_name("fakeQuantizeOnWeights");

    const std::shared_ptr<ov::op::v1::Convolution> convolution = std::make_shared<ov::op::v1::Convolution>(
        concat->output(0),
        fakeQuantizeOnWeights,
        ov::Strides{ 1ul, 1ul },
        ov::CoordinateDiff{ 0, 0 },
        ov::CoordinateDiff{ 0, 0 },
        ov::Strides{ 1ul, 1ul });
    convolution->set_friendly_name("convolution");

    ov::ResultVector results {
        std::make_shared<ov::op::v0::Result>(concat),
        std::make_shared<ov::op::v0::Result>(convolution),
        std::make_shared<ov::op::v0::Result>(fakeQuantize2)
    };

    function = std::make_shared<ov::Model>(results, ov::ParameterVector { input1, input2 }, "OutputLayersHandling");
}

TEST_P(OutputLayersConcat, CompareWithRefImpl) {
    run();
};

}  // namespace LayerTestsDefinitions
