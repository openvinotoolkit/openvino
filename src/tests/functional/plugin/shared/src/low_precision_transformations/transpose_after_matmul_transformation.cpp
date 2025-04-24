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
    ov::element::Type netPrecision;
    ov::PartialShape inputShapes;
    std::string targetDevice;
    ov::pass::low_precision::LayerTransformation::Params params;
    bool perTensor;
    bool transposeChannelDim;
    std::tie(netPrecision, inputShapes, targetDevice, params, perTensor, transposeChannelDim) = obj.param;

    std::ostringstream result;
    result << netPrecision << "_" << targetDevice << "_" << to_string(params) <<
           (perTensor ? "_perTensor" : "_perChannel") <<
        (transposeChannelDim ? "_transposeChannelDim" : "_notTransposeChannelDim");
    return result.str();
}

void TransposeAfterMatMulTransformation::SetUp() {
    ov::element::Type precision;
    ov::PartialShape inputShape;
    ov::pass::low_precision::LayerTransformation::Params params;
    bool perTensor;
    bool transposeChannelDim;
    std::tie(precision, inputShape, targetDevice, params, perTensor, transposeChannelDim) = this->GetParam();

    init_input_shapes({ inputShape, inputShape });

    function = ov::builder::subgraph::TransposeAfterMatMulFunction::getOriginal(precision, inputShape);
}

TEST_P(TransposeAfterMatMulTransformation, CompareWithRefImpl) {
    run();
};

}  // namespace LayerTestsDefinitions
