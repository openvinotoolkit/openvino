// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/space_to_batch_transformation.hpp"

#include <memory>
#include <tuple>

#include "transformations/init_node_info.hpp"
#include "ov_lpt_models/space_to_batch.hpp"

namespace LayerTestsDefinitions {

std::string SpaceToBatchTransformation::getTestCaseName(const testing::TestParamInfo<SpaceToBatchTransformationParams>& obj) {
    const auto& [input_type, target_device, param] = obj.param;

    std::ostringstream result;
    result << input_type << "_" << target_device << "_" << param.input_shape << "_" << param.fake_quantize;
    return result.str();
}

void SpaceToBatchTransformation::SetUp() {
    const auto& [input_type, _targetDevice, param] = this->GetParam();
    targetDevice = _targetDevice;

    init_input_shapes(param.input_shape);

    function = ov::builder::subgraph::SpaceToBatchFunction::get(
        param.input_shape,
        input_type,
        param.fake_quantize,
        param.block_shape,
        param.pads_begin,
        param.pads_end);
}

void SpaceToBatchTransformation::run() {
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

TEST_P(SpaceToBatchTransformation, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
};

}  // namespace LayerTestsDefinitions
