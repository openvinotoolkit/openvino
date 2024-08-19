// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/slice_transformation.hpp"
#include <sstream>
#include <string>
#include <vector>

#include "ov_lpt_models/slice.hpp"

namespace LayerTestsDefinitions {

inline std::ostream& operator<<(std::ostream& os, const std::vector<int64_t>& values) {
    os << "{ ";
    for (size_t i = 0; i < values.size(); ++i) {
        os << values[i];
        if (i != (values.size() - 1ul)) {
            os << ", ";
        }
    }
    os << " }";
    return os;
}

std::string SliceTransformation::getTestCaseName(const testing::TestParamInfo<SliceTransformationParams>& obj) {
    ov::element::Type netPrecision;
    ov::PartialShape inputShape;
    std::string targetDevice;
    ov::pass::low_precision::LayerTransformation::Params params;
    SliceTransformationParam param;;
    std::tie(netPrecision, inputShape, targetDevice, params, param) = obj.param;

    std::ostringstream result;
    result << get_test_case_name_by_params(netPrecision, inputShape, targetDevice, params) << "_" <<
        param.fakeQuantize << "_" <<
        param.start << "_" <<
        param.stop << "_" <<
        param.step << "_" <<
        param.axes;
    return result.str();
}

void SliceTransformation::SetUp() {
    ov::element::Type netPrecision;
    ov::PartialShape inputShape;
    ov::pass::low_precision::LayerTransformation::Params params;
    SliceTransformationParam param;
    std::tie(netPrecision, inputShape, targetDevice, params, param) = this->GetParam();

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

    const auto params = std::get<4>(GetParam());
    const auto& actualPrecision = get_runtime_precision_by_type("StridedSlice");
    EXPECT_EQ(actualPrecision, params.expectedPrecision);
};

} // namespace LayerTestsDefinitions
