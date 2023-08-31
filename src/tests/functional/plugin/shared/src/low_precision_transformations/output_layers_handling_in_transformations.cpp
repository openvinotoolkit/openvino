// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/output_layers.hpp"

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

std::string OutputLayers::getTestCaseName(const testing::TestParamInfo<LayerTestsUtils::LayerTransformationParams>& obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes;
    std::string targetDevice;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    std::tie(netPrecision, inputShapes, targetDevice, params) = obj.param;

    return getTestCaseNameByParams(netPrecision, inputShapes, targetDevice, params);
}

ov::test::utils::InputsMap OutputLayers::get_input_map() {
    auto generate_default = [](const std::shared_ptr<ngraph::Node>& node,
                               size_t port,
                               const ov::element::Type& elemType,
                               const ov::Shape& targetShape) -> ov::runtime::Tensor {
        const double low = 0.0;
        const double high = 255.0;
        return ov::test::utils::create_and_fill_tensor(elemType, targetShape, static_cast<uint32_t>(high - low), low);
    };

    static ov::test::utils::InputsMap inputs_map{{ov::op::Op::get_type_info_static(), generate_default}};
    return inputs_map;
}

void OutputLayers::SetUp() {
    InferenceEngine::SizeVector inputShape;
    InferenceEngine::Precision netPrecision;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    std::tie(netPrecision, inputShape, targetDevice, params) = this->GetParam();

    init_input_shapes(ov::PartialShape(inputShape));

    auto ngPrecision = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    const auto input = std::make_shared<ngraph::opset1::Parameter>(ngPrecision, ngraph::Shape(inputShape));
    input->set_friendly_name("input");

    const float k = 1.f;
    const auto fakeQuantizeOnActivations = ngraph::builder::makeFakeQuantize(
        input->output(0), ngPrecision, 256ul, { 1ul },
        { 0.f }, { 255.f / k }, { 0.f }, { 255.f / k });
    fakeQuantizeOnActivations->set_friendly_name("fakeQuantizeOnActivations");

    const auto weights = ngraph::opset1::Constant::create(
        ngPrecision,
        ngraph::Shape{ inputShape[1ul], inputShape[1ul], 1ul, 1ul },
        std::vector<float>(inputShape[1ul] * inputShape[1ul], 1ul));
    weights->set_friendly_name("weights");
    const auto fakeQuantizeOnWeights = ngraph::builder::makeFakeQuantize(
        weights, ngPrecision, 256ul, { 1ul },
        { -128.f / k }, { 127.f / k }, { -128.f / k }, { 127.f / k });
    fakeQuantizeOnWeights->set_friendly_name("fakeQuantizeOnWeights");

    std::shared_ptr<ngraph::opset1::Convolution> convolution = std::make_shared<ngraph::opset1::Convolution>(
        fakeQuantizeOnActivations,
        fakeQuantizeOnWeights,
        ngraph::Strides{ 1ul, 1ul },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::Strides{ 1ul, 1ul });
    convolution->set_friendly_name("convolution");

    ngraph::ResultVector results {
        std::make_shared<ngraph::opset1::Result>(convolution),
        std::make_shared<ngraph::opset1::Result>(fakeQuantizeOnActivations)
    };

    function = std::make_shared<ngraph::Function>(results, ngraph::ParameterVector { input }, "OutputLayersHandling");
}

TEST_P(OutputLayers, CompareWithRefImpl) {
    run();
};

}  // namespace LayerTestsDefinitions
