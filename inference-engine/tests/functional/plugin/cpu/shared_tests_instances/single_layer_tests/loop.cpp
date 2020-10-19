// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <ngraph/op/util/attr_types.hpp>
#include "single_layer_tests/loop.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
    // without clip values increase rapidly, so use only seq_lenghts = 2
    std::vector<bool> execute_first_iteration{true};
    std::vector<bool> is_body_condition_const{true, false};
    std::vector<bool> body_condition{true, false}; // works only if is_body_condition_const == true
    std::vector<int64_t> trip_count{1, 10, -1}; // -1 means infinity
    std::vector<std::vector<std::pair<std::vector<size_t>, LOOP_IN_TYPE>>> inputs = {
            {{{32, 1, 10}, LOOP_IN_TYPE::INVARIANT}, {{32, 1, 10}, LOOP_IN_TYPE::INVARIANT}, {{32, 1, 10}, LOOP_IN_TYPE::MERGED}},
    };
    std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                             InferenceEngine::Precision::FP16};

    INSTANTIATE_TEST_CASE_P(smoke_LoopCommonZeroClip, LoopTest,
                            ::testing::Combine(
                                    ::testing::ValuesIn(execute_first_iteration),
                                    ::testing::ValuesIn(is_body_condition_const),
                                    ::testing::ValuesIn(body_condition),
                                    ::testing::ValuesIn(trip_count),
                                    ::testing::ValuesIn(inputs),
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                            LoopTest::getTestCaseName);
}  // namespace
