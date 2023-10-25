// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/fake_quantize_precision_selection_transformation.hpp"
#include "ov_lpt_models/fake_quantize_precision_selection.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <ie_core.hpp>

#include <transformations/init_node_info.hpp>

namespace LayerTestsDefinitions {

std::string FakeQuantizePrecisionSelectionTransformation::getTestCaseName(const testing::TestParamInfo<FakeQuantizeTransformationParams>& obj) {
    ngraph::element::Type netPrecision;
    ngraph::PartialShape inputShape;
    std::string targetDevice;
    ov::pass::low_precision::LayerTransformation::Params params;
    FakeQuantizePrecisionSelectionTransformationTestValues testValues;
    std::tie(netPrecision, inputShape, targetDevice, params, testValues) = obj.param;

    std::ostringstream result;
    result << getTestCaseNameByParams(netPrecision, inputShape, targetDevice, params) << "_" << testValues;
    return result.str();
}

void FakeQuantizePrecisionSelectionTransformation::SetUp() {
    ngraph::element::Type netPrecision;
    ngraph::PartialShape inputShape;
    ov::pass::low_precision::LayerTransformation::Params params;
    FakeQuantizePrecisionSelectionTransformationTestValues testValues;
    std::tie(netPrecision, inputShape, targetDevice, params, testValues) = this->GetParam();

    function = ngraph::builder::subgraph::FakeQuantizePrecisionSelectionFunction::getOriginal(
        netPrecision,
        inputShape,
        {
            testValues.operationBeforeLimitedOperationIsPrecisionTransparent,
            testValues.actual.fakeQuantizeOnData,
            testValues.actual.fakeQuantizeOnWeights
        });

    ov::pass::InitNodeInfo().run_on_model(function);
}

TEST_P(FakeQuantizePrecisionSelectionTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
