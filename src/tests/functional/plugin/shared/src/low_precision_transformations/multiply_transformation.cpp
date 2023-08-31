// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/multiply_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <ie_core.hpp>
#include <transformations/init_node_info.hpp>

#include "lpt_ngraph_functions/multiply_function.hpp"
#include "ngraph_functions/subgraph_builders.hpp"


namespace LayerTestsDefinitions {

std::string MultiplyTransformation::getTestCaseName(const testing::TestParamInfo<MultiplyTransformationParams>& obj) {
    ngraph::element::Type precision;
    ngraph::PartialShape inputShapes;
    std::string targetDevice;
    auto params = LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8();
    MultiplyTestValues param;
    std::tie(precision, inputShapes, targetDevice, param) = obj.param;

    std::ostringstream result;
    result << getTestCaseNameByParams(precision, inputShapes, targetDevice, params) <<
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
    ngraph::element::Type precision;
    ngraph::PartialShape inputShape;
    MultiplyTestValues param;
    std::tie(precision, inputShape, targetDevice, param) = this->GetParam();

    auto inputShape1 = inputShape;
    if (param.broadcast1) {
        inputShape1[2] = 1;
        inputShape1[3] = 1;
    }

    ngraph::PartialShape inputShape2;
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

    function = ngraph::builder::subgraph::MultiplyFunction::getOriginal(
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

    auto to_string = [](const ngraph::element::Type& precision) -> std::string {
        switch (precision) {
            case ngraph::element::f32: {
                return "FP32";
            }
            case ngraph::element::i8: {
                return "I8";
            }
            case ngraph::element::u8: {
                return "U8";
            }
            default: {
                return "";
            }
        }
    };

    const auto expectedFqPrecision = to_string(params.expectedPrecisions);
    const auto actualFqPrecision = getRuntimePrecision("multiply");
    EXPECT_EQ(expectedFqPrecision, actualFqPrecision);
}

TEST_P(MultiplyTransformation, CompareWithRefImpl) {
    run();
};

}  // namespace LayerTestsDefinitions
