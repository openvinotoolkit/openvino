// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/fuse_fake_quantize_and_scale_shift_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <ie_core.hpp>

#include <transformations/init_node_info.hpp>

namespace LayerTestsDefinitions {

std::string FuseFakeQuantizeAndScaleShiftTransformation::getTestCaseName(testing::TestParamInfo<FuseFakeQuantizeAndScaleShiftTransformationParams> obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes;
    std::string targetDevice;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantizeOnData;
    std::tie(netPrecision, inputShapes, targetDevice, params, fakeQuantizeOnData) = obj.param;

    std::ostringstream result;
    result << netPrecision << "_" << targetDevice << "_" << fakeQuantizeOnData;
    return result.str();
}

void FuseFakeQuantizeAndScaleShiftTransformation::SetUp() {
    InferenceEngine::SizeVector inputShape;
    InferenceEngine::Precision netPrecision;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantizeOnData;
    std::tie(netPrecision, inputShape, targetDevice, params, fakeQuantizeOnData) = this->GetParam();

    function = ngraph::builder::subgraph::FuseFakeQuantizeAndScaleShiftFunction::getOriginal(
        FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision),
        inputShape,
        fakeQuantizeOnData);

    ngraph::pass::InitNodeInfo().run_on_function(function);

    EXPECT_EQ(1ul, function->get_output_size());
    EXPECT_EQ(1ul, function->get_output_op(0)->get_input_size());
}

TEST_P(FuseFakeQuantizeAndScaleShiftTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
