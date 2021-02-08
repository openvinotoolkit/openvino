// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/fuse_fake_quantize_transformation.hpp"

#include <tuple>
#include <sstream>
#include <string>
#include <vector>

#include <transformations/init_node_info.hpp>
#include "lpt_ngraph_functions/fuse_fake_quantize_function.hpp"

namespace LayerTestsDefinitions {

std::string FuseFakeQuantizeTransformation::getTestCaseName(testing::TestParamInfo<FuseFakeQuantizeTransformationParams> obj) {
    std::string targetDevice;
    FuseFakeQuantizeTransformationTestValues testValues;
    std::tie(targetDevice, testValues) = obj.param;

    std::ostringstream result;
    result << targetDevice << "_" <<
        testValues.actual.precisionBeforeAdd << "_" <<
        testValues.actual.add.values.size() << "_" <<
        testValues.actual.add.outPrecision << "_" <<
        testValues.actual.add.constantShape << "_" <<
        testValues.actual.precisionBeforeDequantization << "_" <<
        testValues.actual.dequantization << "_" <<
        testValues.actual.precisionAfterDequantization << "_" <<
        testValues.actual.fakeQuantizeOnData;
    return result.str();
}

void FuseFakeQuantizeTransformation::SetUp() {
    FuseFakeQuantizeTransformationTestValues testValues;
    std::tie(targetDevice, testValues) = this->GetParam();

    function = ngraph::builder::subgraph::FuseFakeQuantizeFunction::getOriginal(
        testValues.inputShape,
        testValues.actual.precisionBeforeAdd,
        testValues.actual.add,
        testValues.actual.precisionBeforeDequantization,
        testValues.actual.dequantization,
        testValues.actual.precisionAfterDequantization,
        testValues.actual.precisionAfterDequantization,
        testValues.actual.fakeQuantizeOnData);

    ngraph::pass::InitNodeInfo().run_on_function(function);
    validate();
}

void FuseFakeQuantizeTransformation::validate() {
    std::string targetDevice;
    FuseFakeQuantizeTransformationTestValues testValues;
    std::tie(targetDevice, testValues) = this->GetParam();

    const auto transformed = transformNGraph(testValues.params, getLowPrecisionTransformationsNGraph(testValues.params));
    EXPECT_EQ(1ul, transformed->get_output_size());

    const auto output = transformed->get_output_op(0);
    const auto fakeQuantize = output->get_input_node_shared_ptr(0);
    const std::string typeName = fakeQuantize->get_type_name();
    ASSERT_EQ("FakeQuantize", typeName);
}

TEST_P(FuseFakeQuantizeTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
