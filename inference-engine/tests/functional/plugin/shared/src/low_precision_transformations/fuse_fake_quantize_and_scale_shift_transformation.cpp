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
    ngraph::element::Type netPrecision;
    ngraph::Shape inputShape;
    std::string targetDevice;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantizeOnData;
    std::tie(netPrecision, inputShape, targetDevice, params, fakeQuantizeOnData) = obj.param;

    std::ostringstream result;
    result << netPrecision << "_" << targetDevice << "_" << fakeQuantizeOnData;
    return result.str();
}

void FuseFakeQuantizeAndScaleShiftTransformation::SetUp() {
    ngraph::element::Type netPrecision;
    ngraph::Shape inputShape;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantizeOnData;
    std::tie(netPrecision, inputShape, targetDevice, params, fakeQuantizeOnData) = this->GetParam();

    function = ngraph::builder::subgraph::FuseFakeQuantizeAndScaleShiftFunction::getOriginal(
        netPrecision,
        inputShape,
        fakeQuantizeOnData);

    ngraph::pass::InitNodeInfo().run_on_function(function);
    validate();
}

void FuseFakeQuantizeAndScaleShiftTransformation::validate() {
    ngraph::element::Type netPrecision;
    ngraph::Shape inputShape;
    std::string targetDevice;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantizeOnData;
    std::tie(netPrecision, inputShape, targetDevice, params, fakeQuantizeOnData) = this->GetParam();

    const auto transformed = transformNGraph(params, getLowPrecisionTransformationsNGraph(params));
    EXPECT_EQ(1ul, transformed->get_output_size());
    EXPECT_EQ(1ul, function->get_output_op(0)->get_input_size());

    const auto output = transformed->get_output_op(0);
    const auto fakeQuantize = output->get_input_node_shared_ptr(0);
    const std::string typeName = fakeQuantize->get_type_name();
    ASSERT_EQ("FakeQuantize", typeName);
}

TEST_P(FuseFakeQuantizeAndScaleShiftTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
