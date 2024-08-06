// Copyright (C) 2018-2024 Intel Corporation
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
    ov::element::Type netPrecision;
    std::vector<ov::PartialShape> activationsShape;
    std::vector<ov::Shape> weightsShape;
    std::string targetDevice;
    RecurrentCellTransformationParam param;
    ov::pass::low_precision::LayerTransformation::Params params;
    bool addPrecisionTransparentOperations;
    std::tie(netPrecision, activationsShape, weightsShape, targetDevice, params, addPrecisionTransparentOperations, param) = obj.param;

    std::ostringstream result;
    result << get_test_case_name_by_params(netPrecision, activationsShape[0], targetDevice, params) <<
           "FQ_X_" << param.fakeQuantize_X << "_" <<
        "DQ_X_" << param.dequantization_X << "_" <<
        "FQ_W_" << param.fakeQuantize_W << "_" <<
        "DQ_W_" << param.dequantization_W << "_" <<
        "PTO" << addPrecisionTransparentOperations;
    return result.str();
}

void RecurrentCellTransformation::SetUp() {
    ov::element::Type precision;
    std::vector<ov::PartialShape> activations_shapes;
    std::vector<ov::Shape> weights_shapes;
    RecurrentCellTransformationParam param;
    bool addPrecisionTransparentOperations;
    ov::pass::low_precision::LayerTransformation::Params params;

    std::tie(precision, activations_shapes, weights_shapes, targetDevice, params, addPrecisionTransparentOperations, param) = this->GetParam();

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

    const auto params = std::get<6>(GetParam());
    const auto actualPrecision = get_runtime_precision_by_type(params.layerName);
    auto expectedPrecision = params.expectedKernelType;
    if (expectedPrecision == "FP32" && std::get<0>(GetParam()) == ov::element::f16) {
        expectedPrecision = "FP16";
    }
    EXPECT_EQ(actualPrecision, expectedPrecision);
}

TEST_P(RecurrentCellTransformation, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
};

}  // namespace LayerTestsDefinitions
