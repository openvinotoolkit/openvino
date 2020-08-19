// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/split_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <ie_core.hpp>

#include <transformations/init_node_info.hpp>
#include "ngraph_functions/low_precision_transformations/split_function.hpp"

namespace LayerTestsDefinitions {
std::string SplitTransformation::getTestCaseName(testing::TestParamInfo<SplitTransformationParams> obj) {
    ngraph::element::Type netPrecision;
    ngraph::Shape  inputShapes;
    std::string targetDevice;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    SplitTransformationParam param;
    std::tie(netPrecision, inputShapes, targetDevice, params, version, param) = obj.param;

    std::ostringstream result;
    result << getTestCaseNameByParams(netPrecision, inputShapes, targetDevice, params, version) << "_" <<
        param.fakeQuantize << "_axis=" << param.splitedAxis << "_n_splits=" << param.numSplit;
    return result.str();
}

InferenceEngine::Blob::Ptr SplitTransformation::GenerateInput(const InferenceEngine::InputInfo& info) const {
    ngraph::element::Type precision;
    ngraph::Shape inputShape;
    std::string targetDevice;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    SplitTransformationParam param;
    std::tie(precision, inputShape, targetDevice, params, version, param) = this->GetParam();
    const auto& fqOnData = param.fakeQuantize;

    return FuncTestUtils::createAndFillBlobConsistently(
        info.getTensorDesc(),
        static_cast<uint32_t>(fqOnData.empty() ? 25.f : fqOnData.outputHighValues[0] - fqOnData.outputLowValues[0]),
        static_cast<int32_t>(fqOnData.empty() ? -12.5f : fqOnData.outputLowValues[0]),
        1ul);
}

void SplitTransformation::SetUp() {
    ngraph::element::Type precision;
    ngraph::Shape  inputShape;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    SplitTransformationParam param;
    std::tie(precision, inputShape, targetDevice, params, version, param) = this->GetParam();

    ConfigurePlugin(version);

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
