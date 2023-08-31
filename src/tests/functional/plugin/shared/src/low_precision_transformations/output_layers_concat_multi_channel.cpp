// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/output_layers_concat_multi_channel.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>

#include <ie_core.hpp>

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"

#include "ngraph_functions/pass/convert_prc.hpp"
#include "ngraph_functions/builders.hpp"

namespace LayerTestsDefinitions {

std::pair<float, float> outputLayersHandlingInTransformationsForConcatMultiChannelGetInterval(const std::vector<ngraph::element::Type>& precisions) {
    const bool unsignedInterval = std::find(precisions.begin(), precisions.end(), ngraph::element::u8) != precisions.end();
    const float low = unsignedInterval ? 0.f : -128.f;
    const float hight = unsignedInterval ? 255.f : 127.f;
    return std::make_pair(low, hight);
}

std::string OutputLayersConcatMultiChannel::getTestCaseName(
    const testing::TestParamInfo<LayerTestsUtils::LayerTransformationParams>& obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes;
    std::string targetDevice;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    std::tie(netPrecision, inputShapes, targetDevice, params) = obj.param;

    return getTestCaseNameByParams(netPrecision, inputShapes, targetDevice, params);
}

ov::test::utils::InputsMap OutputLayersConcatMultiChannel::get_input_map() {
    auto generate_default = [](const std::shared_ptr<ngraph::Node>&node,
                               size_t port,
                               const ov::element::Type & elemType,
                               const ov::Shape & targetShape) -> ov::runtime::Tensor {
        const auto name = node->get_friendly_name();
        if ((name != "fakeQuantize1") && (name != "fakeQuantize2")) {
            OPENVINO_THROW("unknown name: " + name);
        }
        const double k = (name == "fakeQuantize1") ? 1.f : 2.f;
        const auto interval = outputLayersHandlingInTransformationsForConcatMultiChannelGetInterval({ ngraph::element::u8, ngraph::element::i8 });
        const double low = interval.first / k;
        const double high = interval.second / k;
        const int seed = 1;
        return ov::test::utils::create_and_fill_tensor(elemType, targetShape, static_cast<uint32_t>(high - low), low);
    };

    static ov::test::utils::InputsMap inputs_map{
        { ov::op::Op::get_type_info_static(), generate_default }
    };
    return inputs_map;
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
    rel_threshold = 0.05;

    InferenceEngine::SizeVector inputShape1;
    InferenceEngine::Precision netPrecision;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    std::tie(netPrecision, inputShape1, targetDevice, params) = this->GetParam();

    const InferenceEngine::SizeVector inputShape2 = { inputShape1[0], inputShape1[1] * 2ul, inputShape1[2], inputShape1[3] };
    init_input_shapes({ov::PartialShape(inputShape1), ov::PartialShape(inputShape1)});

    auto ngPrecision = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    const auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngPrecision, ngraph::Shape(inputShape1));
    input1->set_friendly_name("input1");

    const auto fakeQuantize1 = ngraph::builder::makeFakeQuantize(input1->output(0), ngPrecision, 256ul, { 1ul });
    fakeQuantize1->set_friendly_name("fakeQuantize1");

    ASSERT_EQ(4ul, inputShape1.size()) << "unexpected input layout";

    const auto input2 = std::make_shared<ngraph::opset1::Parameter>(ngPrecision, ngraph::Shape(inputShape2));
    input2->set_friendly_name("input2");

    const auto fakeQuantize2 = ngraph::builder::makeFakeQuantize(input2->output(0), ngPrecision, 256ul, { 1ul });
    fakeQuantize2->set_friendly_name("fakeQuantize2");

    const std::shared_ptr<ngraph::opset1::Concat> concat = std::make_shared<ngraph::opset1::Concat>(
        ngraph::OutputVector{ fakeQuantize1->output(0), fakeQuantize2->output(0)}, 1);
    concat->set_friendly_name("concat");

    auto const1 = ngraph::opset1::Constant::create(ngPrecision, ngraph::Shape{ 1, 1, 1, 1 }, { 1 });
    std::shared_ptr<ngraph::opset1::Add> convolution = std::make_shared<ngraph::opset1::Add>(concat, const1);
    convolution->set_friendly_name("convolution");

    ngraph::ResultVector results {
        std::make_shared<ngraph::opset1::Result>(concat),
        std::make_shared<ngraph::opset1::Result>(convolution),
        std::make_shared<ngraph::opset1::Result>(fakeQuantize2)
    };

    function = std::make_shared<ngraph::Function>(results, ngraph::ParameterVector { input1, input2 }, "OutputLayersHandling");
}

TEST_P(OutputLayersConcatMultiChannel, CompareWithRefImpl) {
    run();
};

}  // namespace LayerTestsDefinitions
