// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/fake_quantize_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <ie_core.hpp>

#include <transformations/init_node_info.hpp>

#include "low_precision/fuse_subtract_to_fake_quantize.hpp"
#include "low_precision/fuse_multiply_to_fake_quantize.hpp"

namespace LayerTestsDefinitions {

std::string FakeQuantizeTransformation::getTestCaseName(testing::TestParamInfo<FakeQuantizeTransformationParams> obj) {
    ngraph::element::Type netPrecision;
    ngraph::Shape inputShape;
    std::string targetDevice;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantizeOnData;
    std::tie(netPrecision, inputShape, targetDevice, params, fakeQuantizeOnData) = obj.param;

    std::ostringstream result;
    result << getTestCaseNameByParams(netPrecision, inputShape, targetDevice, params) << "_" << fakeQuantizeOnData;
    return result.str();
}

void FakeQuantizeTransformation::SetUp() {
    ngraph::element::Type netPrecision;
    ngraph::Shape inputShape;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantizeOnData;
    std::tie(netPrecision, inputShape, targetDevice, params, fakeQuantizeOnData) = this->GetParam();

    function = ngraph::builder::subgraph::FakeQuantizeFunction::getOriginal(
        netPrecision,
        inputShape,
        fakeQuantizeOnData);

    ngraph::pass::InitNodeInfo().run_on_function(function);
    validate();
}

void FakeQuantizeTransformation::validate() {
    ngraph::element::Type precision;
    ngraph::Shape inputShapes;
    std::string targetDevice;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantizeOnData;
    std::tie(precision, inputShapes, targetDevice, params, fakeQuantizeOnData) = this->GetParam();

    auto transformations = getLowPrecisionTransformationsNGraph(params);
    transformations.removeStandaloneCleanup<ngraph::pass::low_precision::FuseSubtractToFakeQuantizeTransformation, ngraph::opset1::Subtract>();
    transformations.removeStandaloneCleanup<ngraph::pass::low_precision::FuseMultiplyToFakeQuantizeTransformation, ngraph::opset1::Multiply>();

    const auto transformed = transformNGraph(params, transformations);
    EXPECT_EQ(1ul, transformed->get_output_size());

    const auto output = transformed->get_output_op(0);
    const auto scaleShift = output->get_input_node_shared_ptr(0);
    const std::string typeName = scaleShift->get_type_name();
    ASSERT_EQ("ScaleShiftIE", typeName);
}

TEST_P(FakeQuantizeTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
