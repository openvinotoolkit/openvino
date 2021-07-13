// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/strided_slice_transformation.hpp"
#include <sstream>
#include <string>
#include <vector>
#include <ngraph/ngraph.hpp>

#include "lpt_ngraph_functions/strided_slice_function.hpp"

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

std::string StridedSliceTransformation::getTestCaseName(testing::TestParamInfo<StridedSliceTransformationParams> obj) {
    ngraph::element::Type netPrecision;
    ngraph::PartialShape inputShape;
    std::string targetDevice;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    StridedSliceTransformationParam param;;
    std::tie(netPrecision, inputShape, targetDevice, params, param) = obj.param;

    std::ostringstream result;
    result << getTestCaseNameByParams(netPrecision, inputShape, targetDevice, params) << "_" <<
        param.fakeQuantize << "_" << param.begin << "_" << param.beginMask << "_" <<
        param.end << "_" << param.endMask << "_" << param.strides << "_" << param.newAxisMask <<
        param.shrinkAxisMask << "_" << param.elipsisMask;
    return result.str();
}

void StridedSliceTransformation::SetUp() {
    ngraph::element::Type netPrecision;
    ngraph::PartialShape inputShape;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    StridedSliceTransformationParam param;
    std::tie(netPrecision, inputShape, targetDevice, params, param) = this->GetParam();

    function = ngraph::builder::subgraph::StridedSliceFunction::getOriginal(
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
    Run();
};

} // namespace LayerTestsDefinitions
