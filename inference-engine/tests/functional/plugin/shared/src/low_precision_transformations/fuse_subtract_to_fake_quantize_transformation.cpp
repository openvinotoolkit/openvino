// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/fuse_subtract_to_fake_quantize_transformation.hpp"

#include <tuple>
#include <sstream>
#include <string>
#include <vector>

#include <transformations/init_node_info.hpp>
#include "lpt_ngraph_functions/fuse_subtract_to_fake_quantize_function.hpp"

namespace LayerTestsDefinitions {

std::string FuseSubtractToFakeQuantizeTransformation::getTestCaseName(testing::TestParamInfo<FuseSubtractToFakeQuantizeTransformationParams> obj) {
    std::string targetDevice;
    FuseSubtractToFakeQuantizeTransformationTestValues testValues;
    std::tie(targetDevice, testValues) = obj.param;

    std::ostringstream result;
    result << targetDevice << "_" <<
        testValues.actual.dequantization << "_" <<
        testValues.actual.fakeQuantizeOnData;
    return result.str();
}

void FuseSubtractToFakeQuantizeTransformation::SetUp() {
    FuseSubtractToFakeQuantizeTransformationTestValues testValues;
    std::tie(targetDevice, testValues) = this->GetParam();

    function = ngraph::builder::subgraph::FuseSubtractToFakeQuantizeFunction::get(
        testValues.inputShape,
        testValues.actual.fakeQuantizeOnData,
        testValues.actual.dequantization);

    ngraph::pass::InitNodeInfo().run_on_function(function);
    validate();
}

void FuseSubtractToFakeQuantizeTransformation::validate() {
    std::string targetDevice;
    FuseSubtractToFakeQuantizeTransformationTestValues testValues;
    std::tie(targetDevice, testValues) = this->GetParam();

    const auto transformed = transformNGraph(testValues.params, getLowPrecisionTransformationsNGraph(testValues.params));
    EXPECT_EQ(1ul, transformed->get_output_size());

    const auto output = transformed->get_output_op(0);
    const auto fakeQuantize = output->get_input_node_shared_ptr(0);
    const std::string typeName = fakeQuantize->get_type_name();
    ASSERT_EQ("FakeQuantize", typeName);
}

TEST_P(FuseSubtractToFakeQuantizeTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
