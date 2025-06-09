// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/move_fake_quantize_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>


#include "common_test_utils/common_utils.hpp"
#include "ov_lpt_models/move_fake_quantize.hpp"

namespace LayerTestsDefinitions {

std::string MoveFakeQuantizeTransformation::getTestCaseName(testing::TestParamInfo<MoveFakeQuantizeTransformationParams> obj) {
    auto [netPrecision, inputShape, device, oneInputWithSplit, param] = obj.param;
    std::ostringstream result;
    result << get_test_case_name_by_params(netPrecision, inputShape[0], device) <<
           "SPLIT:" << oneInputWithSplit << "_" <<
        "OP:" << param.operation << "_" <<
        "FQ:" << param.fakeQuantizeAfter << "_" <<
        "DQ:" << param.dequantizationAfter;
    return result.str();
}

void MoveFakeQuantizeTransformation::SetUp() {
    auto [netPrecision, inputShapes, device, oneInputWithSplit, param] = this->GetParam();
    targetDevice = device;

    if (oneInputWithSplit) {
        auto newInputShape = inputShapes[0];
        int channels = 0;
        bool channelsWasIdentified = false;
        for (const auto inputShape : inputShapes) {
            if (inputShape[param.axis].is_static()) {
                channels += inputShape[param.axis].get_length();
                channelsWasIdentified = true;
            }
        }

        if (channelsWasIdentified) {
            newInputShape[param.axis] = channels;
        }
        init_input_shapes(newInputShape);
    } else {
        init_input_shapes(inputShapes);
    }

    function = ov::builder::subgraph::MoveFakeQuantize::get(
        netPrecision,
        inputShapes,
        param.concatInputsCount,
        {},
        {},
        {},
        param.operation,
        param.fakeQuantizeAfter,
        param.convertAfter,
        param.dequantizationAfter,
        {},
        {},
        param.axis,
        oneInputWithSplit);
}

void MoveFakeQuantizeTransformation::run() {
    LayerTransformation::run();

    const auto params = std::get<4>(GetParam());
    const auto actualPrecision = get_runtime_precision_by_type(params.layerName);
    auto expectedPrecision = params.expectedKernelType;
    if (expectedPrecision == "f32" && std::get<0>(GetParam()) == ov::element::f16) {
        expectedPrecision = "f16";
    }
    EXPECT_EQ(actualPrecision, expectedPrecision);
}

TEST_P(MoveFakeQuantizeTransformation, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
};

}  // namespace LayerTestsDefinitions
