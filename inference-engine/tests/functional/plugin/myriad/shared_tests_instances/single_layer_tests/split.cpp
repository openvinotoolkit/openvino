// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/split.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP16
};

INSTANTIATE_TEST_CASE_P(smoke_NumSplitsCheck, SplitLayerTest,
                        ::testing::Combine(
                                ::testing::Values(1),
                                // TODO: 0-axis excluded
                                //  Check (status == ie::StatusCode::OK) failed: Failed to reshape Network:
                                //  Failed to infer shapes for Split layer (Split_2) with error:
                                //  The sum of the dimensions on the axis(0) is not equal out_sizes: [30]
                                ::testing::Values(1, 2, 3),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(std::vector<size_t >({30, 30, 30, 30})),
                                ::testing::Values(CommonTestUtils::DEVICE_MYRIAD)),
                        SplitLayerTest::getTestCaseName);

}  // namespace
