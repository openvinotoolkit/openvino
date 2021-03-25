// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/convolution_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>

#include <ie_core.hpp>

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "ngraph_functions/pass/convert_prc.hpp"
#include "lpt_ngraph_functions/fake_quantize_and_convolution_function.hpp"

namespace LayerTestsDefinitions {

std::string ConvolutionTransformation::getTestCaseName(testing::TestParamInfo<ConvolutionTransformationParams> obj) {
    ngraph::element::Type netPrecision;
    ngraph::Shape inputShape;
    std::string targetDevice;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    ConvolutionTransformationParam param;
    std::tie(netPrecision, inputShape, targetDevice, params, param) = obj.param;

    std::ostringstream result;
    result << getTestCaseNameByParams(netPrecision, inputShape, targetDevice, params) << "_" <<
        param.fakeQuantizeOnData << "_" <<
        param.fakeQuantizeOnWeights << "_" <<
        param.add.constantShape;
    return result.str();
}

void ConvolutionTransformation::SetUp() {
    threshold = 0.1f;

    ngraph::element::Type netPrecision;
    ngraph::Shape inputShape;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    ConvolutionTransformationParam param;
    std::tie(netPrecision, inputShape, targetDevice, params, param) = this->GetParam();

    function = ngraph::builder::subgraph::FakeQuantizeAndConvolutionFunction::get(
        netPrecision,
        inputShape,
        // TODO: pass from test parameters
        param.fakeQuantizeOnData,
        param.fakeQuantizeOnWeights,
        param.add);

    validate();
}

void ConvolutionTransformation::Run() {
    LayerTestsCommon::Run();

    const auto params = std::get<4>(GetParam());
    const auto actualType = getRuntimePrecision(params.layerName);
    EXPECT_EQ(actualType, params.expectedKernelType);
}

void ConvolutionTransformation::validate() {
    ngraph::element::Type netPrecision;
    ngraph::Shape inputShape;
    std::string targetDevice;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    ConvolutionTransformationParam param;
    std::tie(netPrecision, inputShape, targetDevice, params, param) = this->GetParam();

    const auto transformed = transformNGraph(params, getLowPrecisionTransformationsNGraph(params));
    EXPECT_EQ(1ul, transformed->get_output_size());

    const auto output = transformed->get_output_op(0);
    const auto parent = output->get_input_node_shared_ptr(0);
    ASSERT_FALSE(parent == nullptr);

    const std::string typeName = parent->get_type_name();
    const auto isQuantizationSupported = [](const ngraph::builder::subgraph::FakeQuantizeOnData& fq) {
        return (fq.quantizationLevel == 255) || (fq.quantizationLevel == 256);
    };

    if (param.fakeQuantizeOnData.empty() || (!isQuantizationSupported(param.fakeQuantizeOnData)) ||
        param.fakeQuantizeOnWeights.empty() || (!isQuantizationSupported(param.fakeQuantizeOnWeights))) {
        ASSERT_EQ("ConvolutionIE", typeName);
    } else {
        ASSERT_EQ("ScaleShiftIE", typeName);
    }
}

TEST_P(ConvolutionTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
