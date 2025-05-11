// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/multiply_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include "transformations/init_node_info.hpp"

#include "ov_lpt_models/multiply_partial_function.hpp"


namespace LayerTestsDefinitions {

std::string MultiplyTransformation::getTestCaseName(const testing::TestParamInfo<MultiplyTransformationParams>& obj) {
    ov::element::Type precision;
    ov::PartialShape inputShapes;
    std::string targetDevice;
    auto params = LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8();
    MultiplyTestValues param;
    std::tie(precision, inputShapes, targetDevice, param) = obj.param;

    std::ostringstream result;
    result << get_test_case_name_by_params(precision, inputShapes, targetDevice, params) <<
           (param.broadcast1 ? "_broadcast1" : "") <<
        (param.broadcast2 ? "_broadcast2" : "");

    result << "_" << param.expectedPrecisions << "_";

    if (!param.fakeQuantize1.empty()) {
        result << "_on_branch1_" <<
            param.fakeQuantize1.inputLowValues[0] << "_" <<
            param.fakeQuantize1.inputHighValues[0] << "_" <<
            param.fakeQuantize1.outputLowValues[0] << "_" <<
            param.fakeQuantize1.outputHighValues[0];
    }
    if (!param.fakeQuantize2.empty()) {
        result << "_on_branch2_" <<
            param.fakeQuantize2.inputLowValues[0] << "_" <<
            param.fakeQuantize2.inputHighValues[0] << "_" <<
            param.fakeQuantize2.outputLowValues[0] << "_" <<
            param.fakeQuantize2.outputHighValues[0];
    }
    result << "_" << param.secondInputIsConstant;
    return result.str();
}

void MultiplyTransformation::SetUp() {
    ov::element::Type precision;
    ov::PartialShape inputShape;
    MultiplyTestValues param;
    std::tie(precision, inputShape, targetDevice, param) = this->GetParam();

    auto inputShape1 = inputShape;
    if (param.broadcast1) {
        inputShape1[2] = 1;
        inputShape1[3] = 1;
    }

    ov::PartialShape inputShape2;
    if (param.secondInputIsConstant) {
        inputShape2 = {};
    } else {
        inputShape2 = inputShape;
        if (param.broadcast2) {
            inputShape2[2] = 1;
            inputShape2[3] = 1;
        }
    }
    init_input_shapes(
            param.secondInputIsConstant ?
            std::vector<ov::PartialShape>{ inputShape1 } :
            std::vector<ov::PartialShape>{ inputShape1, inputShape2 });

    function = ov::builder::subgraph::MultiplyPartialFunction::get(
        precision,
        inputShape,
        param.broadcast1,
        param.fakeQuantize1,
        param.broadcast2,
        param.fakeQuantize2,
        param.fakeQuantizeAfter,
        param.secondInputIsConstant);

    ov::pass::InitNodeInfo().run_on_model(function);
}

void MultiplyTransformation::run() {
    LayerTransformation::run();

    const auto params = std::get<3>(GetParam());

    auto to_string = [](const ov::element::Type& precision) -> std::string {
        switch (precision) {
            case ov::element::f32: {
                return "f32";
            }
            case ov::element::i8: {
                return "i8";
            }
            case ov::element::u8: {
                return "u8";
            }
            default: {
                return "";
            }
        }
    };

    const auto expectedFqPrecision = to_string(params.expectedPrecisions);
    const auto actualFqPrecision = get_runtime_precision("multiply");
    EXPECT_EQ(expectedFqPrecision, actualFqPrecision);
}

TEST_P(MultiplyTransformation, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
};

}  // namespace LayerTestsDefinitions
