// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/prelu_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <ie_core.hpp>

#include <transformations/init_node_info.hpp>
#include "ov_lpt_models/prelu.hpp"

namespace LayerTestsDefinitions {

std::string PReluTransformation::getTestCaseName(const testing::TestParamInfo<PReluTransformationParams>& obj) {
    ngraph::element::Type precision;
    ngraph::PartialShape inputShape;
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
    ngraph::PartialShape inputShape;
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
    ngraph::PartialShape inputShape;
    PReluTestValues testValues;
    std::tie(precision, inputShape, targetDevice, testValues) = this->GetParam();

    function = ngraph::builder::subgraph::PReluFunction::getOriginal(inputShape, precision, testValues.fakeQuantize);

    ov::pass::InitNodeInfo().run_on_model(function);
}

TEST_P(PReluTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
