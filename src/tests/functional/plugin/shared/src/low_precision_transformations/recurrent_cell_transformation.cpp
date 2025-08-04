// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/recurrent_cell_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>

#include "common_test_utils/common_utils.hpp"
#include "ov_lpt_models/recurrent_cell.hpp"

namespace LayerTestsDefinitions {

std::string RecurrentCellTransformation::getTestCaseName(testing::TestParamInfo<RecurrentCellTransformationParams> obj) {
    auto [netPrecision, activationsShape, weightsShape, device, addPrecisionTransparentOperations, param] = obj.param;

    std::ostringstream result;
    result << get_test_case_name_by_params(netPrecision, activationsShape[0], device) <<
           "FQ_X_" << param.fakeQuantize_X << "_" <<
           "DQ_X_" << param.dequantization_X << "_" <<
           "FQ_W_" << param.fakeQuantize_W << "_" <<
           "DQ_W_" << param.dequantization_W << "_" <<
           "PTO" << addPrecisionTransparentOperations;
    return result.str();
}

void RecurrentCellTransformation::SetUp() {
    auto [precision, activations_shapes, weights_shapes, device, addPrecisionTransparentOperations, param] = this->GetParam();
    targetDevice = device;

    init_input_shapes(activations_shapes);

    function = ov::builder::subgraph::RecurrentCellFunction::get(precision,
                                                                 activations_shapes,
                                                                 weights_shapes,
                                                                 param.RNNType,
                                                                 {
                                                                     param.fakeQuantize_X,
                                                                     param.fakeQuantize_H,
                                                                     param.fakeQuantize_W,
                                                                     param.fakeQuantize_R
                                                                 },
                                                                 {
                                                                     param.convert_X,
                                                                     param.convert_H,
                                                                     param.convert_W,
                                                                     param.convert_R
                                                                 },
                                                                 {
                                                                     param.dequantization_X,
                                                                     param.dequantization_H,
                                                                     param.dequantization_W,
                                                                     param.dequantization_R
                                                                 },
                                                                 addPrecisionTransparentOperations);
}

void RecurrentCellTransformation::run() {
    LayerTransformation::run();

    const auto params = std::get<5>(GetParam());
    const auto actualPrecision = get_runtime_precision_by_type(params.layerName);
    auto expectedPrecision = params.expectedKernelType;
    if (expectedPrecision == "f32" && std::get<0>(GetParam()) == ov::element::f16) {
        expectedPrecision = "f16";
    }
    EXPECT_EQ(actualPrecision, expectedPrecision);
}

TEST_P(RecurrentCellTransformation, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
};

}  // namespace LayerTestsDefinitions
