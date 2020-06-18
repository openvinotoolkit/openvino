// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/reduce_ops.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
};

const std::vector<std::vector<size_t>> inputShapes = {
        std::vector<size_t>{10, 20, 30, 40},
};

const std::vector<std::vector<int>> axes = {
        {0},
        {0, 3},
        {1, -1},
};
const std::vector<ngraph::helpers::ReductionType> reductionTypes = {
        ngraph::helpers::ReductionType::Mean,
        ngraph::helpers::ReductionType::Min,
        ngraph::helpers::ReductionType::Max,
        ngraph::helpers::ReductionType::Sum,
        ngraph::helpers::ReductionType::Prod,
        ngraph::helpers::ReductionType::LogicalOr,
        ngraph::helpers::ReductionType::LogicalXor,
        ngraph::helpers::ReductionType::LogicalAnd,
};

const auto params = testing::Combine(
        testing::ValuesIn(axes),
        testing::Values(true, false),
        testing::ValuesIn(reductionTypes),
        testing::ValuesIn(netPrecisions),
        testing::ValuesIn(inputShapes),
        testing::Values(CommonTestUtils::DEVICE_CPU)
);


INSTANTIATE_TEST_CASE_P(
        Reduce,
        ReduceOpsLayerTest,
        params,
        ReduceOpsLayerTest::getTestCaseName
);
}  // namespace
