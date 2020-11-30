// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/group_convolution_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>

#include <ie_core.hpp>

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/layer_test_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "ngraph_functions/pass/convert_prc.hpp"
#include "lpt_ngraph_functions/group_convolution_function.hpp"

namespace LayerTestsDefinitions {

std::string GroupConvolutionTransformation::getTestCaseName(testing::TestParamInfo<GroupConvolutionTransformationParams> obj) {
    ngraph::element::Type netPrecision;
    std::string targetDevice;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    GroupConvolutionTransformationParam param;
    std::tie(netPrecision, targetDevice, params, param) = obj.param;

    std::ostringstream result;
    result <<
        getTestCaseNameByParams(netPrecision, param.inputShape, targetDevice, params) << "_" <<
        param.inputShape << "_" <<
        param.outputShape << "_" <<
        param.group << "_" <<
        param.fakeQuantizeOnData << "_" <<
        param.fakeQuantizeOnWeights;
    return result.str();
}

void GroupConvolutionTransformation::SetUp() {
    threshold = 0.1f;

    ngraph::element::Type netPrecision;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    GroupConvolutionTransformationParam param;
    std::tie(netPrecision, targetDevice, params, param) = this->GetParam();

    function = ngraph::builder::subgraph::GroupConvolutionFunction::getOriginal(
        netPrecision,
        param.inputShape,
        param.outputShape,
        param.group,
        param.fakeQuantizeOnData,
        param.fakeQuantizeOnWeights);

    validateNGraph();
}

void GroupConvolutionTransformation::validateNGraph() {
    ngraph::element::Type netPrecision;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    GroupConvolutionTransformationParam param;

    std::tie(netPrecision, targetDevice, params, param) = this->GetParam();

    auto transformed = transformNGraph(params);
    EXPECT_EQ(1ul, transformed->get_output_size());
    std::shared_ptr<ngraph::Node> output = transformed->get_output_op(0);

    std::shared_ptr<ngraph::Node> parent = output->get_input_node_shared_ptr(0);
    ASSERT_FALSE(parent == nullptr);
    const std::string typeName = parent->get_type_name();

    ASSERT_TRUE(typeName == "ScaleShiftIE" || typeName == "PowerIE" || typeName == "ConvolutionIE");
}

TEST_P(GroupConvolutionTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
