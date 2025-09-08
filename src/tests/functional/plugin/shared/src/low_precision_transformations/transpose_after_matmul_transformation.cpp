// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/transpose_after_matmul_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>


#include "common_test_utils/common_utils.hpp"

#include "ov_lpt_models/transpose_after_mat_mul.hpp"


namespace LayerTestsDefinitions {

std::string TransposeAfterMatMulTransformation::getTestCaseName(const testing::TestParamInfo<TransposeAfterMatMulTransformationParams>& obj) {
    auto [netPrecision, inputShapes, device, perTensor, transposeChannelDim] = obj.param;
    std::ostringstream result;
    result << netPrecision << "_" << device <<
           (perTensor ? "_perTensor" : "_perChannel") <<
           (transposeChannelDim ? "_transposeChannelDim" : "_notTransposeChannelDim");
    return result.str();
}

void TransposeAfterMatMulTransformation::SetUp() {
    auto [precision, inputShape, device, perTensor, transposeChannelDim] = this->GetParam();
    targetDevice = device;

    init_input_shapes({ inputShape, inputShape });

    function = ov::builder::subgraph::TransposeAfterMatMulFunction::getOriginal(precision, inputShape);
}

TEST_P(TransposeAfterMatMulTransformation, CompareWithRefImpl) {
    run();
};

}  // namespace LayerTestsDefinitions
