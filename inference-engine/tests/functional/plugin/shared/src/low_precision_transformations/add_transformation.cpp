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

std::string AddTransformation::getTestCaseName(testing::TestParamInfo<AddTransformationParams> obj) {
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
        (param.broadcast ? "_broadcast" : "") <<
        param.fakeQuantize1 << "_" <<
        param.operation1 << "_" <<
        param.fakeQuantize2 << "_" <<
        param.operation2;

    for (const auto& expected : param.expected) {
        result << "_" << expected;
    }
    return result.str();
}

void AddTransformation::SetUp() {
    ngraph::element::Type precision;
    ngraph::Shape inputShape;
    AddTestValues param;
    std::tie(precision, inputShape, targetDevice, param) = this->GetParam();

    function = ngraph::builder::subgraph::AddFunction::getOriginal(
        precision,
        inputShape,
        param.broadcast,
        param.fakeQuantize1,
        param.operation1,
        param.fakeQuantize2,
        param.operation2,
        param.constInput);

    ngraph::pass::VisualizeTree("c:\\Projects\\temp\\test.original").run_on_function(function);

    ngraph::pass::InitNodeInfo().run_on_function(function);
    validate();
}

void AddTransformation::validate() {
    ngraph::element::Type precision;
    ngraph::Shape inputShape;
    std::string targetDevice;
    AddTestValues param;
    std::tie(precision, inputShape, targetDevice, param) = this->GetParam();

    const auto params = LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8();
    const auto transformed = transformNGraph(params, getLowPrecisionTransformationsNGraph(params));

    const auto output = transformed->get_output_op(0);
    if ((!param.fakeQuantize1.empty()) && (!param.fakeQuantize2.empty())) {
        const auto scaleShift = output->get_input_node_shared_ptr(0);
        const std::string typeName = scaleShift->get_type_name();
        ASSERT_EQ("ScaleShiftIE", typeName);
    }
}

TEST_P(AddTransformation, CompareWithRefImpl) {
    Run();

    const auto params = std::get<3>(GetParam());
    for (const auto& expected : params.expected) {
        const LayerTestsCommon::PerformanceItem actual = getPerformanceItem(expected.name);
        EXPECT_EQ(expected.type, actual.layerType);
        EXPECT_EQ(expected.expectedKernelType, actual.runtimePrecision);
    }
};

}  // namespace LayerTestsDefinitions
