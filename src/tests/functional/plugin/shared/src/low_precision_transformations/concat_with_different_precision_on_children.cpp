// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/concat_with_different_precision_on_children.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>

#include "transformations/init_node_info.hpp"
#include "ov_lpt_models/concat.hpp"

namespace LayerTestsDefinitions {

std::string ConcatWithDifferentChildrenTransformation::getTestCaseName(const testing::TestParamInfo<ConcatWithDifferentChildrenTransformationParams>& obj) {
    auto [netPrecision, inputShapes, device, param] = obj.param;

    std::ostringstream result;
    result <<
           get_test_case_name_by_params(netPrecision, inputShapes, device) <<
           "_axis_" << param.axis << "_" << param.fqOnData1 << param.fqOnData2;

    return result.str();
}

void ConcatWithDifferentChildrenTransformation::SetUp() {
    auto [netPrecision, inputShapes, device, param] = this->GetParam();
    targetDevice = device;

    init_input_shapes({ inputShapes, inputShapes });

    function = ov::builder::subgraph::ConcatFunction::getOriginalWithDifferentPrecisionOnChildren(
        netPrecision, inputShapes, param.axis, param.fqOnData1, param.fqOnData2);
}

TEST_P(ConcatWithDifferentChildrenTransformation, CompareWithRefImpl) {
    run();
};

}  // namespace LayerTestsDefinitions
