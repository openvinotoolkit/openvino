// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/transpose_after_matmul_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>

#include <ie_core.hpp>

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"

#include "lpt_ngraph_functions/transpose_after_mat_mul_function.hpp"


namespace LayerTestsDefinitions {

std::string TransposeAfterMatMulTransformation::getTestCaseName(testing::TestParamInfo<TransposeAfterMatMulTransformationParams> obj) {
    ngraph::element::Type netPrecision;
    ngraph::PartialShape inputShapes;
    std::string targetDevice;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    bool perTensor;
    bool transposeChannelDim;
    std::tie(netPrecision, inputShapes, targetDevice, params, perTensor, transposeChannelDim) = obj.param;

    std::ostringstream result;
    result << netPrecision << "_" << targetDevice << "_" << toString(params) <<
        (perTensor ? "_perTensor" : "_perChannel") <<
        (transposeChannelDim ? "_transposeChannelDim" : "_notTransposeChannelDim");
    return result.str();
}

void TransposeAfterMatMulTransformation::SetUp() {
    ngraph::element::Type precision;
    ngraph::PartialShape inputShape;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    bool perTensor;
    bool transposeChannelDim;
    std::tie(precision, inputShape, targetDevice, params, perTensor, transposeChannelDim) = this->GetParam();

    function = ngraph::builder::subgraph::TransposeAfterMatMulFunction::getOriginal(precision, inputShape);
}

TEST_P(TransposeAfterMatMulTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
