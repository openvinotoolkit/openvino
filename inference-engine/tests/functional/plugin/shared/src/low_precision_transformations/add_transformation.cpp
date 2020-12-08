// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/add_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <ie_core.hpp>

#include <transformations/init_node_info.hpp>
#include "lpt_ngraph_functions/add_function.hpp"
#include "ngraph_functions/subgraph_builders.hpp"

namespace LayerTestsDefinitions {

std::string AddTransformation::getTestCaseName(testing::TestParamInfo< AddTransformationParams> obj) {
    ngraph::element::Type netPrecision;
    ngraph::Shape inputShapes;
    std::string targetDevice;
    auto params = LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8();
    AddTestValues param;
    std::tie(netPrecision, inputShapes, targetDevice, param) = obj.param;

    if (!param.precisionOnActivations.empty()) {
        params.precisionsOnActivations = param.precisionOnActivations;
    }

    std::ostringstream result;
    result << getTestCaseNameByParams(netPrecision, inputShapes, targetDevice, params) <<
        (param.broadcast ? "_broadcast" : "");
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
    return result.str();
}

void AddTransformation::SetUp() {
    ngraph::element::Type precision;
    ngraph::Shape inputShape1;
    AddTestValues param;
    std::tie(precision, inputShape1, targetDevice, param) = this->GetParam();

    ngraph::Shape inputShape2 = inputShape1;

    if (param.broadcast) {
        inputShape2[2] = 1;
        inputShape2[3] = 1;
    }

    auto fq1 = param.fakeQuantize1;
    auto fq2 = param.fakeQuantize2;

    const auto input1 = std::make_shared<ngraph::opset1::Parameter>(precision, inputShape1);
    const auto fakeQuantize1 = fq1.empty() ?
        nullptr :
        ngraph::builder::makeFakeQuantize(
            input1, precision, fq1.quantizationLevel, fq1.constantShape,
            fq1.inputLowValues, fq1.inputHighValues, fq1.outputLowValues, fq1.outputHighValues);

    const auto input2 = std::make_shared<ngraph::opset1::Parameter>(precision, inputShape2);
    const auto fakeQuantize2 = fq2.empty() ?
        nullptr :
        ngraph::builder::makeFakeQuantize(
            input2, precision, fq2.quantizationLevel, fq2.constantShape,
            fq2.inputLowValues, fq2.inputHighValues, fq2.outputLowValues, fq2.outputHighValues);

    const auto add = std::make_shared<ngraph::opset1::Add>(
        fq1.empty() ? input1 : fakeQuantize1,
        fq2.empty() ? input2 : fakeQuantize2);

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(add) };
    function = std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input1, input2 }, "AddTransformation");

    ngraph::pass::InitNodeInfo().run_on_function(function);
}

TEST_P(AddTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
