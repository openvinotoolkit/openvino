// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/multiply_with_one_parent_transformation.hpp"

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "common_test_utils/common_utils.hpp"
#include "ov_lpt_models/multiply_with_one_parent.hpp"

namespace LayerTestsDefinitions {

std::string MultiplyWithOneParentTransformation::getTestCaseName(const testing::TestParamInfo<MultiplyWithOneParentTransformationParams>& obj) {
    ov::element::Type netPrecision;
    ov::PartialShape inputShape;
    std::string targetDevice;
    MultiplyWithOneParentTransformationValues values;

    std::tie(netPrecision, inputShape, targetDevice, values) = obj.param;

    std::ostringstream result;
    result << netPrecision << "_" << inputShape;
    return result.str();
}

void MultiplyWithOneParentTransformation::SetUp() {
    ov::element::Type netPrecision;
    ov::PartialShape inputShape;
    ov::pass::low_precision::LayerTransformation::Params params;
    MultiplyWithOneParentTransformationValues values;
    std::tie(netPrecision, inputShape, targetDevice, values) = this->GetParam();

    init_input_shapes(inputShape);

    function = ov::builder::subgraph::MultiplyWithOneParentFunction::getOriginal(netPrecision, inputShape, values.fakeQuantize);
}

TEST_P(MultiplyWithOneParentTransformation, CompareWithRefImpl) {
    run();
};

}  // namespace LayerTestsDefinitions
