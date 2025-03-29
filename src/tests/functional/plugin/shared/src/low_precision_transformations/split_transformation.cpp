// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/split_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>

#include "transformations/init_node_info.hpp"
#include "low_precision/split.hpp"
#include "ov_lpt_models/split.hpp"

namespace LayerTestsDefinitions {
std::string SplitTransformation::getTestCaseName(const testing::TestParamInfo<SplitTransformationParams>& obj) {
    ov::element::Type netPrecision;
    ov::PartialShape  inputShapes;
    std::string targetDevice;
    ov::pass::low_precision::LayerTransformation::Params params;
    SplitTransformationParam param;
    std::tie(netPrecision, inputShapes, targetDevice, params, param) = obj.param;

    std::ostringstream result;
    result << get_test_case_name_by_params(netPrecision, inputShapes, targetDevice, params) << "_" <<
           param.fakeQuantize << "_axis=" << param.splitedAxis << "_n_splits=" << param.numSplit;
    return result.str();
}


void SplitTransformation::SetUp() {
    ov::element::Type precision;
    ov::PartialShape inputShape;
    ov::pass::low_precision::LayerTransformation::Params params;
    SplitTransformationParam param;
    std::tie(precision, inputShape, targetDevice, params, param) = this->GetParam();

    init_input_shapes(inputShape);

    function = ov::builder::subgraph::SplitFunction::getOriginal(
        precision,
        inputShape,
        param.fakeQuantize,
        param.splitedAxis,
        param.numSplit);
}

TEST_P(SplitTransformation, CompareWithRefImpl) {
    run();
};
} // namespace LayerTestsDefinitions
