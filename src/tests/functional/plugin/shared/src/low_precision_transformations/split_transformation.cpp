// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/split_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <ie_core.hpp>

#include <transformations/init_node_info.hpp>
#include "low_precision/split.hpp"
#include "ov_lpt_models/split.hpp"

namespace LayerTestsDefinitions {
std::string SplitTransformation::getTestCaseName(const testing::TestParamInfo<SplitTransformationParams>& obj) {
    ngraph::element::Type netPrecision;
    ngraph::PartialShape  inputShapes;
    std::string targetDevice;
    ov::pass::low_precision::LayerTransformation::Params params;
    SplitTransformationParam param;
    std::tie(netPrecision, inputShapes, targetDevice, params, param) = obj.param;

    std::ostringstream result;
    result << getTestCaseNameByParams(netPrecision, inputShapes, targetDevice, params) << "_" <<
        param.fakeQuantize << "_axis=" << param.splitedAxis << "_n_splits=" << param.numSplit;
    return result.str();
}

InferenceEngine::Blob::Ptr SplitTransformation::GenerateInput(const InferenceEngine::InputInfo& info) const {
    ngraph::element::Type precision;
    ngraph::PartialShape inputShape;
    std::string targetDevice;
    ov::pass::low_precision::LayerTransformation::Params params;
    SplitTransformationParam param;
    std::tie(precision, inputShape, targetDevice, params, param) = this->GetParam();
    const auto& fqOnData = param.fakeQuantize;

    return FuncTestUtils::createAndFillBlobConsistently(
        info.getTensorDesc(),
        static_cast<uint32_t>(fqOnData.empty() ? 25.f : fqOnData.outputHighValues[0] - fqOnData.outputLowValues[0]),
        static_cast<int32_t>(fqOnData.empty() ? -12.5f : fqOnData.outputLowValues[0]),
        1ul);
}

void SplitTransformation::SetUp() {
    ngraph::element::Type precision;
    ngraph::PartialShape inputShape;
    ov::pass::low_precision::LayerTransformation::Params params;
    SplitTransformationParam param;
    std::tie(precision, inputShape, targetDevice, params, param) = this->GetParam();

    function = ngraph::builder::subgraph::SplitFunction::getOriginal(
        precision,
        inputShape,
        param.fakeQuantize,
        param.splitedAxis,
        param.numSplit);
}

TEST_P(SplitTransformation, CompareWithRefImpl) {
    Run();
};
} // namespace LayerTestsDefinitions
