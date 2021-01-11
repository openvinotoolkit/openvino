// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/prelu_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <ie_core.hpp>

#include <transformations/init_node_info.hpp>
#include "lpt_ngraph_functions/prelu_function.hpp"

namespace LayerTestsDefinitions {

std::string PReluTransformation::getTestCaseName(testing::TestParamInfo<PReluTransformationParams> obj) {
    ngraph::element::Type precision;
    ngraph::Shape inputShape;
    std::string targetDevice;
    PReluTestValues testValues;
    std::tie(precision, inputShape, targetDevice, testValues) = obj.param;

    std::ostringstream result;
    result <<
        precision << "_" <<
        targetDevice << "_" <<
        testValues.fakeQuantize;

    return result.str();
}

InferenceEngine::Blob::Ptr PReluTransformation::GenerateInput(const InferenceEngine::InputInfo &info) const {
    ngraph::element::Type precision;
    ngraph::Shape inputShape;
    std::string targetDevice;
    PReluTestValues testValues;
    std::tie(precision, inputShape, targetDevice, testValues) = this->GetParam();

    const auto fqOnData = testValues.fakeQuantize;
    return FuncTestUtils::createAndFillBlobConsistently(
        info.getTensorDesc(),
        static_cast<uint32_t>(fqOnData.empty() ? 25.f : fqOnData.outputHighValues[0] - fqOnData.outputLowValues[0]),
        static_cast<int32_t>(fqOnData.empty() ? -12.5f : fqOnData.outputLowValues[0]),
        1ul);
}

void PReluTransformation::SetUp() {
    ngraph::element::Type precision;
    ngraph::Shape inputShape;
    PReluTestValues testValues;
    std::tie(precision, inputShape, targetDevice, testValues) = this->GetParam();

    function = ngraph::builder::subgraph::PReluFunction::getOriginal(inputShape, precision, testValues.fakeQuantize);

    ngraph::pass::InitNodeInfo().run_on_function(function);
    validate();
}

void PReluTransformation::validate() {
    ngraph::element::Type precision;
    ngraph::Shape inputShape;
    std::string targetDevice;
    PReluTestValues testValues;
    std::tie(precision, inputShape, targetDevice, testValues) = this->GetParam();

    auto params = LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8();
    const auto transformed = transformNGraph(params, getLowPrecisionTransformationsNGraph(params));

    const auto output = transformed->get_output_op(0);
    const auto layer = output->get_input_node_shared_ptr(0);
    const std::string typeName = layer->get_type_name();
    if ((!testValues.fakeQuantize.empty()) && (!testValues.isSubtract)) {
        ASSERT_EQ("ScaleShiftIE", typeName);
    } else {
        ASSERT_EQ("ReLUIE", typeName);
    }
}

TEST_P(PReluTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
