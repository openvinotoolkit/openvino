// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/fake_quantize_precision_selection_transformation.hpp"
#include "lpt_ngraph_functions/fake_quantize_precision_selection_function.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <ie_core.hpp>

#include <transformations/init_node_info.hpp>

namespace LayerTestsDefinitions {

std::string FakeQuantizePrecisionSelectionTransformation::getTestCaseName(testing::TestParamInfo<FakeQuantizeTransformationParams> obj) {
    ngraph::element::Type netPrecision;
    ngraph::Shape inputShape;
    std::string targetDevice;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    FakeQuantizePrecisionSelectionTransformationTestValues testValues;
    std::tie(netPrecision, inputShape, targetDevice, params, testValues) = obj.param;

    std::ostringstream result;
    result << getTestCaseNameByParams(netPrecision, inputShape, targetDevice, params) << "_" << testValues;
    return result.str();
}

void FakeQuantizePrecisionSelectionTransformation::SetUp() {
    ngraph::element::Type netPrecision;
    ngraph::Shape inputShape;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    FakeQuantizePrecisionSelectionTransformationTestValues testValues;
    std::tie(netPrecision, inputShape, targetDevice, params, testValues) = this->GetParam();

    function = ngraph::builder::subgraph::FakeQuantizePrecisionSelectionFunction::getOriginal(
        netPrecision,
        inputShape,
        {
            testValues.operationBeforeLimitedOperationIsPrecisionTransparent,
            testValues.actual.fakeQuantizeOnData,
            testValues.actual.fakeQuantizeOnWeights
        });

    ngraph::pass::InitNodeInfo().run_on_function(function);
    validate();
}

void FakeQuantizePrecisionSelectionTransformation::validate() {
    ngraph::element::Type precision;
    ngraph::Shape inputShapes;
    std::string targetDevice;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    FakeQuantizePrecisionSelectionTransformationTestValues param;
    std::tie(precision, inputShapes, targetDevice, params, param) = this->GetParam();

    const auto transformed = transformNGraph(params, getLowPrecisionTransformationsNGraph(params));
    EXPECT_EQ(1ul, transformed->get_output_size());

    const auto output = transformed->get_output_op(0);
    const auto concat = output->get_input_node_shared_ptr(0);

    const std::string typeName = concat->get_type_name();
    ASSERT_EQ("Concat", typeName);

    EXPECT_EQ(2ul, concat->get_input_size());

    const auto scaleShiftOrConv = concat->get_input_node_shared_ptr(0);
    const std::string scaleShiftOrConvName = scaleShiftOrConv->get_type_name();
    if (param.operationBeforeLimitedOperationIsPrecisionTransparent) {
        ASSERT_EQ("ScaleShiftIE", scaleShiftOrConvName);
    } else {
        ASSERT_EQ("ConvolutionIE", scaleShiftOrConvName);
    }

    const auto scaleShift = concat->get_input_node_shared_ptr(1);
    const std::string scaleShiftName = scaleShift->get_type_name();
    ASSERT_EQ("ScaleShiftIE", scaleShiftName);
}

TEST_P(FakeQuantizePrecisionSelectionTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
