// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/mat_mul_with_optimized_constant_fq.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>


#include "common_test_utils/common_utils.hpp"
#include "ov_lpt_models/mat_mul_with_optimized_constant_fake_quantize.hpp"

namespace LayerTestsDefinitions {

std::string MatMulWithOptimizedConstantFq::getTestCaseName(
    const testing::TestParamInfo<MatMulWithOptimizedConstantFakeQuantizeTransformationTransformationParams>& obj) {
    auto [netPrecision, shapes, device, param] = obj.param;

    std::ostringstream result;
    result << netPrecision << "_" <<
        shapes.first << "_" << shapes.second << "_" <<
        device << "_"  <<
        param.fqOnData << "_" <<
        param.fqOnWeights;
    return result.str();
}

void MatMulWithOptimizedConstantFq::SetUp() {
    auto [precision, shapes, device, param] = this->GetParam();
    targetDevice = device;

    init_input_shapes({ shapes.first, shapes.second });

    function = ov::builder::subgraph::MatMulWithOptimizedConstantFakeQuantizeFunction::getOriginal(
        precision,
        shapes.first,
        shapes.second,
        param.fqOnData,
        param.fqOnWeights);
}

TEST_P(MatMulWithOptimizedConstantFq, CompareWithRefImpl) {
    run();
};

}  // namespace LayerTestsDefinitions
