// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/relu_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <ie_core.hpp>

#include <transformations/init_node_info.hpp>
#include "ov_lpt_models/relu.hpp"

namespace LayerTestsDefinitions {

std::string ReluTransformation::getTestCaseName(const testing::TestParamInfo<ReluTransformationParams>& obj) {
    ngraph::element::Type precision;
    ngraph::PartialShape inputShape;
    std::string targetDevice;
    ReluTestValues testValues;
    std::tie(precision, inputShape, targetDevice, testValues) = obj.param;

    std::ostringstream result;
    result <<
        precision << "_" <<
        targetDevice << "_" <<
        testValues.fakeQuantize;

    return result.str();
}

#if 0
InferenceEngine::Blob::Ptr ReluTransformation::GenerateInput(const InferenceEngine::InputInfo &info) const {
    ngraph::element::Type precision;
    ngraph::PartialShape inputShape;
    std::string targetDevice;
    ReluTestValues testValues;
    std::tie(precision, inputShape, targetDevice, testValues) = this->GetParam();

    const auto fqOnData = testValues.fakeQuantize;
    return FuncTestUtils::createAndFillBlobConsistently(
        info.getTensorDesc(),
        static_cast<uint32_t>(fqOnData.empty() ? 25.f : fqOnData.outputHighValues[0] - fqOnData.outputLowValues[0]),
        static_cast<int32_t>(fqOnData.empty() ? -12.5f : fqOnData.outputLowValues[0]),
        1ul);
}
#endif

void ReluTransformation::SetUp() {
    abs_threshold = 1.0;
    ngraph::element::Type precision;
    ngraph::PartialShape inputShape;
    ReluTestValues testValues;
    std::tie(precision, inputShape, targetDevice, testValues) = this->GetParam();

    init_input_shapes(inputShape);

    function = ngraph::builder::subgraph::ReluFunction::getOriginal(inputShape, precision, testValues.fakeQuantize);

    ov::pass::InitNodeInfo().run_on_model(function);
}

TEST_P(ReluTransformation, CompareWithRefImpl) {
    run();
};

}  // namespace LayerTestsDefinitions
