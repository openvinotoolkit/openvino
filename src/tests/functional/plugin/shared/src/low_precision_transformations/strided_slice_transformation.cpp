// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/strided_slice_transformation.hpp"
#include <sstream>
#include <string>
#include <vector>

#include "ov_lpt_models/strided_slice.hpp"

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

std::string StridedSliceTransformation::getTestCaseName(const testing::TestParamInfo<StridedSliceTransformationParams>& obj) {
    ov::element::Type netPrecision;
    ov::PartialShape inputShape;
    std::string targetDevice;
    ov::pass::low_precision::LayerTransformation::Params params;
    StridedSliceTransformationParam param;;
    std::tie(netPrecision, inputShape, targetDevice, params, param) = obj.param;

    std::ostringstream result;
    result << get_test_case_name_by_params(netPrecision, inputShape, targetDevice, params) << "_" <<
           param.fakeQuantize << "_" << param.begin << "_" << param.beginMask << "_" <<
        param.end << "_" << param.endMask << "_" << param.strides << "_" << param.newAxisMask <<
        param.shrinkAxisMask << "_" << param.elipsisMask;
    return result.str();
}

void StridedSliceTransformation::SetUp() {
    ov::element::Type netPrecision;
    ov::PartialShape inputShape;
    ov::pass::low_precision::LayerTransformation::Params params;
    StridedSliceTransformationParam param;
    std::tie(netPrecision, inputShape, targetDevice, params, param) = this->GetParam();

    init_input_shapes(inputShape);

    function = ov::builder::subgraph::StridedSliceFunction::getOriginal(
        netPrecision,
        inputShape,
        param.fakeQuantize,
        param.begin,
        param.end,
        param.strides,
        param.beginMask,
        param.endMask,
        param.newAxisMask,
        param.shrinkAxisMask,
        param.elipsisMask);
}

TEST_P(StridedSliceTransformation, CompareWithRefImpl) {
    run();
};

} // namespace LayerTestsDefinitions
