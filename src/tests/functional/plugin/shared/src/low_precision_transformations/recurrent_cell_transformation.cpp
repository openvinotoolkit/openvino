// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/recurrent_cell_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>

#include <ie_core.hpp>

#include "common_test_utils/common_utils.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "ov_lpt_models/recurrent_cell.hpp"

namespace LayerTestsDefinitions {

std::string RecurrentCellTransformation::getTestCaseName(testing::TestParamInfo<RecurrentCellTransformationParams> obj) {
    ngraph::element::Type netPrecision;
    std::vector<ngraph::PartialShape> activationsShape;
    std::vector<ngraph::Shape> weightsShape;
    std::string targetDevice;
    RecurrentCellTransformationParam param;
    ov::pass::low_precision::LayerTransformation::Params params;
    std::tie(netPrecision, activationsShape, weightsShape, targetDevice, params, param) = obj.param;

    std::ostringstream result;
    result << getTestCaseNameByParams(netPrecision, activationsShape[0], targetDevice, params) <<
        "FQ_X_" << param.fakeQuantize_X << "_" <<
        "DQ_X_" << param.dequantization_X << "_" <<
        "FQ_W_" << param.fakeQuantize_W << "_" <<
        "DQ_W_" << param.dequantization_W;
    return result.str();
}

void RecurrentCellTransformation::SetUp() {
    ngraph::element::Type precision;
    std::vector<ngraph::PartialShape> activations_shapes;
    std::vector<ngraph::Shape> weights_shapes;
    RecurrentCellTransformationParam param;
    ov::pass::low_precision::LayerTransformation::Params params;

    std::tie(precision, activations_shapes, weights_shapes, targetDevice, params, param) = this->GetParam();

    function = ngraph::builder::subgraph::RecurrentCellFunction::get(precision,
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
                                                                      });
}

void RecurrentCellTransformation::Run() {
    LayerTestsCommon::Run();

    if (!executableNetwork)
        return;

    const auto params = std::get<5>(GetParam());
    const auto actualPrecision = getRuntimePrecisionByType(params.layerName);
    auto expectedPrecision = params.expectedKernelType;
    if (expectedPrecision == "FP32" && std::get<0>(GetParam()) == ngraph::element::f16) {
        expectedPrecision = "FP16";
    }
    EXPECT_EQ(actualPrecision, expectedPrecision);
}

TEST_P(RecurrentCellTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
