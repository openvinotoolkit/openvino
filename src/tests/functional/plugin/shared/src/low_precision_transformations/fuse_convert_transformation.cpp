// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/fuse_convert_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>


#include "common_test_utils/common_utils.hpp"

#include "ov_lpt_models/fuse_convert.hpp"

namespace LayerTestsDefinitions {

std::string FuseConvertTransformation::getTestCaseName(const testing::TestParamInfo<FuseConvertTransformationParams>& obj) {
    auto [precision, shape, device, deqOperations, constInput] = obj.param;
    std::ostringstream result;
    result << get_test_case_name_by_params(precision, shape, device) <<
           "_" << deqOperations << "_" << constInput;
    return result.str();
}

void FuseConvertTransformation::SetUp() {
    auto [precision, shape, device, deqOperations, constInput] = this->GetParam();
    targetDevice = device;

    init_input_shapes(constInput ? std::vector<ov::PartialShape>{ shape } : std::vector<ov::PartialShape>{ shape, shape });

    function = ov::builder::subgraph::FuseConvertFunction::getWithFQ(
        shape,
        precision,
        deqOperations,
        constInput);
}

TEST_P(FuseConvertTransformation, CompareWithRefImpl) {
    run();
};

}  // namespace LayerTestsDefinitions
