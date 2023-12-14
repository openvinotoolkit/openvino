// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/variadic_split_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <ie_core.hpp>

#include <transformations/init_node_info.hpp>
#include "low_precision/variadic_split.hpp"
#include "ov_lpt_models/variadic_split.hpp"

namespace LayerTestsDefinitions {
std::string VariadicSplitTransformation::getTestCaseName(const testing::TestParamInfo<VariadicSplitTransformationParams>& obj) {
    ngraph::element::Type netPrecision;
    ngraph::PartialShape inputShapes;
    std::string targetDevice;
    ov::pass::low_precision::LayerTransformation::Params params;
    VariadicSplitTransformationParam param;
    std::tie(netPrecision, inputShapes, targetDevice, params, param) = obj.param;

    std::ostringstream result;
    result << getTestCaseNameByParams(netPrecision, inputShapes, targetDevice, params) << "_" <<
        param.fakeQuantize << "_axis=" << param.splitedAxis << "_splitLengths={ ";
    for (size_t i = 0; i < param.splitLengths.size(); ++i) {
        result << param.splitLengths[i];
        if (i != (param.splitLengths.size() - 1ul)) {
            result << ", ";
        }
    }
    result << " }";
    return result.str();
}

#if 0
InferenceEngine::Blob::Ptr VariadicSplitTransformation::GenerateInput(const InferenceEngine::InputInfo& info) const {
    ngraph::element::Type precision;
    ngraph::PartialShape inputShape;
    std::string targetDevice;
    ov::pass::low_precision::LayerTransformation::Params params;
    VariadicSplitTransformationParam param;
    std::tie(precision, inputShape, targetDevice, params, param) = this->GetParam();
    const auto& fqOnData = param.fakeQuantize;

    return FuncTestUtils::createAndFillBlobConsistently(
        info.getTensorDesc(),
        static_cast<uint32_t>(fqOnData.empty() ? 25.f : fqOnData.outputHighValues[0] - fqOnData.outputLowValues[0]),
        static_cast<int32_t>(fqOnData.empty() ? -12.5f : fqOnData.outputLowValues[0]),
        1ul);
}
#endif

void VariadicSplitTransformation::SetUp() {
    ngraph::element::Type precision;
    ngraph::PartialShape inputShape;
    ov::pass::low_precision::LayerTransformation::Params params;
    VariadicSplitTransformationParam param;
    std::tie(precision, inputShape, targetDevice, params, param) = this->GetParam();

    init_input_shapes(inputShape);

    function = ngraph::builder::subgraph::VariadicSplitFunction::getOriginal(
        precision,
        inputShape,
        param.fakeQuantize,
        param.splitedAxis,
        param.splitLengths);
}

TEST_P(VariadicSplitTransformation, CompareWithRefImpl) {
    run();
};
} // namespace LayerTestsDefinitions
