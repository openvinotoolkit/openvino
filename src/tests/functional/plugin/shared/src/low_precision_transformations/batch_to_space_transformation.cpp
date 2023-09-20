// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/batch_to_space_transformation.hpp"

#include <memory>
#include <tuple>
#include <ie_core.hpp>

#include <transformations/init_node_info.hpp>
#include "ov_lpt_models/batch_to_space.hpp"

namespace LayerTestsDefinitions {

std::string BatchToSpaceTransformation::getTestCaseName(const testing::TestParamInfo<BatchToSpaceTransformationParams>& obj) {
    ngraph::element::Type input_type;
    std::string target_device;
    BatchToSpaceTransformationParam param;
    std::tie(input_type, target_device, param) = obj.param;

    std::ostringstream result;
    result << input_type << "_" << target_device << "_" << param.input_shape << "_" << param.fake_quantize;
    return result.str();
}

void BatchToSpaceTransformation::SetUp() {
    ngraph::element::Type input_type;
    BatchToSpaceTransformationParam param;
    std::tie(input_type, targetDevice, param) = this->GetParam();

    function = ngraph::builder::subgraph::BatchToSpaceFunction::get(
        param.input_shape,
        input_type,
        param.fake_quantize,
        param.block_shape,
        param.crops_begin,
        param.crops_end);
}

void BatchToSpaceTransformation::Run() {
    LayerTestsCommon::Run();

    const auto params = std::get<2>(GetParam());
    auto expected_type = params.expected_kernel_type;
    const auto input_type = std::get<0>(GetParam());
    if ((expected_type == "f32") && (input_type == ov::element::f16)) {
        expected_type = "f16";
    }

    const auto actual_type = getRuntimePrecisionByType(params.layer_type);
    EXPECT_EQ(actual_type, expected_type);
}

TEST_P(BatchToSpaceTransformation, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    Run();
};

}  // namespace LayerTestsDefinitions
