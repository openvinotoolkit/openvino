// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/clamp_transformation.hpp"
#include <sstream>
#include <string>
#include <vector>
#include <ngraph/ngraph.hpp>

#include "lpt_ngraph_functions/clamp_function.hpp"

namespace LayerTestsDefinitions {

std::string ClampTransformation::getTestCaseName(testing::TestParamInfo<ClampTransformationParams> obj) {
    ngraph::element::Type netPrecision;
    ngraph::Shape inputShape;
    std::string targetDevice;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    ClampTransformationParam param;;
    std::tie(netPrecision, inputShape, targetDevice, params, param) = obj.param;

    std::ostringstream result;
    result << getTestCaseNameByParams(netPrecision, inputShape, targetDevice, params) << "_" <<
        param.fakeQuantize << "_" <<
        "min=" << param.clampLowConst <<
        "max=" << param.clampHighConst;
    return result.str();
}

void ClampTransformation::SetUp() {
    ngraph::element::Type netPrecision;
    ngraph::Shape inputShape;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    ClampTransformationParam param;
    std::tie(netPrecision, inputShape, targetDevice, params, param) = this->GetParam();

    function = ngraph::builder::subgraph::ClampFunction::getOriginal(
        netPrecision,
        inputShape,
        param.fakeQuantize,
        param.clampLowConst,
        param.clampHighConst);

    validate();
}

void ClampTransformation::validate() {
    ngraph::element::Type netPrecision;
    ngraph::Shape inputShape;
    std::string targetDevice;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    ClampTransformationParam param;
    std::tie(netPrecision, inputShape, targetDevice, params, param) = this->GetParam();

    const auto transformed = transformNGraph(params, getLowPrecisionTransformationsNGraph(params));

    EXPECT_EQ(1ul, transformed->get_output_size());
    std::shared_ptr<ngraph::Node> output = transformed->get_output_op(0);

    std::shared_ptr<ngraph::Node> parent = output->get_input_node_shared_ptr(0);
    ASSERT_FALSE(parent == nullptr);
    const std::string typeName = parent->get_type_name();
    if (!param.dequantizationAfter.empty()) {
        EXPECT_EQ("ScaleShiftIE", typeName);
        EXPECT_EQ(3, parent->get_input_size());

        const auto expectedScale = param.dequantizationAfter.multiply.values;
        const auto actualScale =
            ngraph::as_type_ptr<ngraph::opset1::Constant>(parent->get_input_node_shared_ptr(1))->cast_vector<float>();
        EXPECT_EQ(expectedScale.size(), actualScale.size());

        const auto expectedShift = param.dequantizationAfter.subtract.values;
        const auto actualShift =
            ngraph::as_type_ptr<ngraph::opset1::Constant>(parent->get_input_node_shared_ptr(2))->cast_vector<float>();
        EXPECT_EQ(expectedShift.size(), actualShift.size());
    }
}

TEST_P(ClampTransformation, CompareWithRefImpl) {
    Run();
};

} // namespace LayerTestsDefinitions
