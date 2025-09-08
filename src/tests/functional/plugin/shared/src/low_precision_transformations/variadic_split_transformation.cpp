// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/variadic_split_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>

#include "transformations/init_node_info.hpp"
#include "low_precision/variadic_split.hpp"
#include "ov_lpt_models/variadic_split.hpp"

namespace LayerTestsDefinitions {
std::string VariadicSplitTransformation::getTestCaseName(const testing::TestParamInfo<VariadicSplitTransformationParams>& obj) {
    auto [netPrecision, inputShapes, device, param] = obj.param;
    std::ostringstream result;
    result << get_test_case_name_by_params(netPrecision, inputShapes, device) << "_" <<
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


void VariadicSplitTransformation::SetUp() {
    auto [precision, inputShape, device, param] = this->GetParam();
    targetDevice = device;

    init_input_shapes(inputShape);

    function = ov::builder::subgraph::VariadicSplitFunction::getOriginal(
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
