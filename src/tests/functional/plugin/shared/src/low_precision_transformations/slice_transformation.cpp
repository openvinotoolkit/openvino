// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/slice_transformation.hpp"
#include <sstream>
#include <string>
#include <vector>

#include "ov_lpt_models/slice.hpp"

namespace LayerTestsDefinitions {

std::string SliceTransformation::getTestCaseName(const testing::TestParamInfo<SliceTransformationParams>& obj) {
    auto [netPrecision, inputShape, device, param] = obj.param;
    std::ostringstream result;
    result << get_test_case_name_by_params(netPrecision, inputShape, device) << "_" <<
        param.fakeQuantize << "_" <<
        ov::test::utils::vec2str(param.start) << "_" <<
        ov::test::utils::vec2str(param.stop) << "_" <<
        ov::test::utils::vec2str(param.step) << "_" <<
        ov::test::utils::vec2str(param.axes);
    return result.str();
}

void SliceTransformation::SetUp() {
    auto [netPrecision, inputShape, device, param] = this->GetParam();
    targetDevice = device;

    init_input_shapes(inputShape);

    function = ov::builder::subgraph::SliceFunction::get(
        netPrecision,
        inputShape,
        param.fakeQuantize,
        param.start,
        param.stop,
        param.step,
        param.axes);
}

TEST_P(SliceTransformation, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();

    const auto params = std::get<3>(GetParam());
    const auto& actualPrecision = get_runtime_precision_by_type("StridedSlice");
    EXPECT_EQ(actualPrecision, params.expectedPrecision);
};

} // namespace LayerTestsDefinitions
