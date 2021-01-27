// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/convolution_with_incorrect_weights.hpp"

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
#include "lpt_ngraph_functions/convolution_function.hpp"

namespace LayerTestsDefinitions {

std::string ConvolutionWIthIncorrectWeightsTransformation::getTestCaseName(testing::TestParamInfo<ConvolutionWIthIncorrectWeightsParams> obj) {
    ngraph::element::Type netPrecision;
    ngraph::Shape inputShape;
    std::string targetDevice;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    ConvolutionWIthIncorrectWeightsParam param;
    std::tie(netPrecision, inputShape, targetDevice, params, param) = obj.param;

    std::ostringstream result;
    result << getTestCaseNameByParams(netPrecision, inputShape, targetDevice, params) <<
        (param.isCorrect ? "_correct_weights" : "_incorrect_weights") <<
        (param.fakeQuantizeOnData.empty() ? "_noFqOnActivations" : "") <<
        (param.fakeQuantizeOnWeights.empty() ? "_noFqOnWeights" : "");
    return result.str();
}

void ConvolutionWIthIncorrectWeightsTransformation::SetUp() {
    threshold = 0.1f;

    ngraph::element::Type netPrecision;
    ngraph::Shape inputShape;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    ConvolutionWIthIncorrectWeightsParam param;
    std::tie(netPrecision, inputShape, targetDevice, params, param) = this->GetParam();

    function = ngraph::builder::subgraph::ConvolutionFunction::getOriginalWithIncorrectWeights(
        inputShape,
        netPrecision,
        param.fakeQuantizeOnWeights,
        param.fakeQuantizeOnData,
        param.isCorrect);

    validate();
}

void ConvolutionWIthIncorrectWeightsTransformation::validate() {
    ngraph::element::Type netPrecision;
    ngraph::Shape inputShape;
    std::string targetDevice;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    ConvolutionWIthIncorrectWeightsParam param;
    std::tie(netPrecision, inputShape, targetDevice, params, param) = this->GetParam();

    const auto transformed = transformNGraph(params, getLowPrecisionTransformationsNGraph(params));
    EXPECT_EQ(1ul, transformed->get_output_size());

    const auto output = transformed->get_output_op(0);
    const auto parent = output->get_input_node_shared_ptr(0);
    ASSERT_FALSE(parent == nullptr);

    const std::string typeName = parent->get_type_name();
    if (param.isCorrect) {
        ASSERT_EQ("ScaleShiftIE", typeName);
    } else {
        ASSERT_EQ("ConvolutionIE", typeName);
    }
}

TEST_P(ConvolutionWIthIncorrectWeightsTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
