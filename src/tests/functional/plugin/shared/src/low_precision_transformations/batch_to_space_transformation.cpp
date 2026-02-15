// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/batch_to_space_transformation.hpp"

#include <memory>
#include <tuple>

#include "transformations/init_node_info.hpp"
#include "ov_lpt_models/batch_to_space.hpp"

namespace LayerTestsDefinitions {

std::string BatchToSpaceTransformation::getTestCaseName(const testing::TestParamInfo<BatchToSpaceTransformationParams>& obj) {
    auto [input_type, device, param] = obj.param;

    std::ostringstream result;
    result << input_type << "_" << device << "_" << param.input_shape << "_" << param.fake_quantize;
    return result.str();
}

void BatchToSpaceTransformation::SetUp() {
    auto [input_type, device, param] = this->GetParam();
    targetDevice = device;

    init_input_shapes(param.input_shape);

    function = ov::builder::subgraph::BatchToSpaceFunction::get(
        param.input_shape,
        input_type,
        param.fake_quantize,
        param.block_shape,
        param.crops_begin,
        param.crops_end);
}

void BatchToSpaceTransformation::run() {
    LayerTransformation::run();

    const auto params = std::get<2>(GetParam());
    auto expected_type = params.expected_kernel_type;
    const auto input_type = std::get<0>(GetParam());
    if ((expected_type == "f32") && (input_type == ov::element::f16)) {
        expected_type = "f16";
    }

    const auto actual_type = get_runtime_precision_by_type(params.layer_type);
    EXPECT_EQ(actual_type, expected_type);
}

TEST_P(BatchToSpaceTransformation, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
};

}  // namespace LayerTestsDefinitions
