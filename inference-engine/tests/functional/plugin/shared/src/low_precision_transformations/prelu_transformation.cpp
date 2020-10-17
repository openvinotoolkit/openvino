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
#include "ngraph_functions/low_precision_transformations/prelu_function.hpp"

namespace LayerTestsDefinitions {

std::string PReluTransformation::getTestCaseName(testing::TestParamInfo<PReluTransformationParams> obj) {
    ngraph::element::Type precision;
    ngraph::Shape inputShape;
    std::string targetDevice;
    ngraph::builder::subgraph::FakeQuantizeOnData fqOnData;
    std::tie(precision, inputShape, targetDevice, fqOnData) = obj.param;

    std::ostringstream result;
    result <<
        precision << "_" <<
        targetDevice << "_" <<
        fqOnData;

    return result.str();
}

InferenceEngine::Blob::Ptr PReluTransformation::GenerateInput(const InferenceEngine::InputInfo &info) const {
    ngraph::element::Type precision;
    ngraph::Shape inputShape;
    std::string targetDevice;
    ngraph::builder::subgraph::FakeQuantizeOnData fqOnData;
    std::tie(precision, inputShape, targetDevice, fqOnData) = this->GetParam();

    return FuncTestUtils::createAndFillBlobConsistently(
        info.getTensorDesc(),
        static_cast<uint32_t>(fqOnData.empty() ? 25.f : fqOnData.outputHighValues[0] - fqOnData.outputLowValues[0]),
        static_cast<int32_t>(fqOnData.empty() ? -12.5f : fqOnData.outputLowValues[0]),
        1ul);
}

void PReluTransformation::SetUp() {
    ngraph::element::Type precision;
    ngraph::Shape inputShape;
    ngraph::builder::subgraph::FakeQuantizeOnData fqOnData;
    std::tie(precision, inputShape, targetDevice, fqOnData) = this->GetParam();

    function = ngraph::builder::subgraph::PReluFunction::getOriginal(inputShape, precision, fqOnData);

    ngraph::pass::InitNodeInfo().run_on_function(function);
}

TEST_P(PReluTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
