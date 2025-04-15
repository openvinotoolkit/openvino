// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/fake_quantize_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>

#include "transformations/init_node_info.hpp"

#include "low_precision/fuse_subtract_to_fake_quantize.hpp"
#include "low_precision/fuse_multiply_to_fake_quantize.hpp"

namespace LayerTestsDefinitions {

std::string FakeQuantizeTransformation::getTestCaseName(const testing::TestParamInfo<FakeQuantizeTransformationParams>& obj) {
    ov::element::Type netPrecision;
    ov::PartialShape inputShape;
    std::string targetDevice;
    ov::pass::low_precision::LayerTransformation::Params params;
    FakeQuantizeTransformationParam testParams;
    bool isConvertOnConstants;
    std::tie(netPrecision, inputShape, targetDevice, params, testParams, isConvertOnConstants) = obj.param;

    std::ostringstream result;
    result << get_test_case_name_by_params(netPrecision, inputShape, targetDevice, params) << "_" <<
           isConvertOnConstants << "_" << testParams.fakequantize;
    return result.str();
}

void FakeQuantizeTransformation::SetUp() {
    ov::element::Type netPrecision;
    ov::PartialShape inputShape;
    ov::pass::low_precision::LayerTransformation::Params params;
    FakeQuantizeTransformationParam testParams;
    bool isConvertOnConstants;
    std::tie(netPrecision, inputShape, targetDevice, params, testParams, isConvertOnConstants) = this->GetParam();

    init_input_shapes(inputShape);

    testParams.fakequantize.addConverts = isConvertOnConstants;

    function = ov::builder::subgraph::FakeQuantizeFunction::getOriginal(
        params,
        netPrecision,
        inputShape,
        testParams.fakequantize,
        true);

    ov::pass::InitNodeInfo().run_on_model(function);
}

void FakeQuantizeTransformation::run() {
    LayerTransformation::run();

    const auto params = std::get<4>(GetParam());
    const auto actualPrecision = get_runtime_precision_by_type(params.layerName);
    auto expectedPrecision = params.expectedKernelType;
    if (expectedPrecision == "FP32" && std::get<0>(GetParam()) == ov::element::f16) {
        expectedPrecision = "FP16";
    }

    EXPECT_EQ(actualPrecision, expectedPrecision);
}

TEST_P(FakeQuantizeTransformation, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
};

}  // namespace LayerTestsDefinitions
