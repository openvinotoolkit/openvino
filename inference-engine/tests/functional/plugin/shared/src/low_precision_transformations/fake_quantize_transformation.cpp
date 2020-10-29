// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/fake_quantize_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <ie_core.hpp>

#include <transformations/init_node_info.hpp>

namespace LayerTestsDefinitions {

std::string FakeQuantizeTransformation::getTestCaseName(testing::TestParamInfo<FakeQuantizeTransformationParams> obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes;
    std::string targetDevice;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantizeOnData;
    std::tie(netPrecision, inputShapes, targetDevice, params, fakeQuantizeOnData) = obj.param;

    std::ostringstream result;
    result << getTestCaseNameByParams(netPrecision, inputShapes, targetDevice, params) << "_" << fakeQuantizeOnData;
    return result.str();
}

void FakeQuantizeTransformation::SetUp() {
    InferenceEngine::SizeVector inputShape;
    InferenceEngine::Precision netPrecision;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantizeOnData;
    std::tie(netPrecision, inputShape, targetDevice, params, fakeQuantizeOnData) = this->GetParam();

    function = ngraph::builder::subgraph::FakeQuantizeFunction::getOriginal(
        FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision),
        inputShape,
        fakeQuantizeOnData);

    ngraph::pass::InitNodeInfo().run_on_function(function);
}

TEST_P(FakeQuantizeTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
