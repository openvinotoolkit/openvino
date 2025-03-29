// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/output_layers_concat_multi_channel.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>


#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/node_builders/fake_quantize.hpp"


namespace LayerTestsDefinitions {

std::pair<float, float> outputLayersHandlingInTransformationsForConcatMultiChannelGetInterval(const std::vector<ov::element::Type>& precisions) {
    const bool unsignedInterval = std::find(precisions.begin(), precisions.end(), ov::element::u8) != precisions.end();
    const float low = unsignedInterval ? 0.f : -128.f;
    const float hight = unsignedInterval ? 255.f : 127.f;
    return std::make_pair(low, hight);
}

std::string OutputLayersConcatMultiChannel::getTestCaseName(
    const testing::TestParamInfo<LayerTestsUtils::LayerTransformationParams>& obj) {
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
*  Convolution/Power  Output
*        /
*       /
*   Output
*/

void OutputLayersConcatMultiChannel::SetUp() {
    ov::Shape inputShape1;
    ov::element::Type ngPrecision;
    ov::pass::low_precision::LayerTransformation::Params params;
    std::tie(ngPrecision, inputShape1, targetDevice, params) = this->GetParam();

    const ov::Shape inputShape2 = { inputShape1[0], inputShape1[1] * 2ul, inputShape1[2], inputShape1[3] };
    init_input_shapes({ov::PartialShape(inputShape1), ov::PartialShape(inputShape1)});

    const auto input1 = std::make_shared<ov::op::v0::Parameter>(ngPrecision, ov::Shape(inputShape1));
    input1->set_friendly_name("input1");

    const auto fakeQuantize1 = ov::test::utils::make_fake_quantize(input1->output(0), ngPrecision, 256ul, { 1ul });
    fakeQuantize1->set_friendly_name("fakeQuantize1");

    ASSERT_EQ(4ul, inputShape1.size()) << "unexpected input layout";
    const auto input2 = std::make_shared<ov::op::v0::Parameter>(ngPrecision, ov::Shape(inputShape2));
    input2->set_friendly_name("input2");

    const auto fakeQuantize2 = ov::test::utils::make_fake_quantize(input2->output(0), ngPrecision, 256ul, { 1ul });
    fakeQuantize2->set_friendly_name("fakeQuantize2");

    const std::shared_ptr<ov::op::v0::Concat> concat = std::make_shared<ov::op::v0::Concat>(
        ov::OutputVector{ fakeQuantize1->output(0), fakeQuantize2->output(0)}, 1);
    concat->set_friendly_name("concat");

    auto const1 = ov::op::v0::Constant::create(ngPrecision, ov::Shape{ 1, 1, 1, 1 }, { 1 });
    std::shared_ptr<ov::op::v1::Add> convolution = std::make_shared<ov::op::v1::Add>(concat, const1);
    convolution->set_friendly_name("convolution");

    ov::ResultVector results {
        std::make_shared<ov::op::v0::Result>(concat),
        std::make_shared<ov::op::v0::Result>(convolution),
        std::make_shared<ov::op::v0::Result>(fakeQuantize2)
    };

    function = std::make_shared<ov::Model>(results, ov::ParameterVector { input1, input2 }, "OutputLayersHandling");
}

TEST_P(OutputLayersConcatMultiChannel, CompareWithRefImpl) {
    run();
};

}  // namespace LayerTestsDefinitions
