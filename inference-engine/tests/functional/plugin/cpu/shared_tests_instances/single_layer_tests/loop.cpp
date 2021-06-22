// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "single_layer_tests/loop.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
    // without clip values increase rapidly, so use only seq_lengths = 2
    std::vector<bool> execute_first_iteration{true};
    std::vector<bool> is_body_condition_const{true/*, false*/};
    std::vector<bool> body_condition{true/*, false*/}; // works only if is_body_condition_const == true
    std::vector<int64_t> trip_count{1, 10/*, -1*/}; // -1 means infinity
    std::vector<std::vector<std::pair<std::vector<size_t>, LOOP_IN_TYPE>>> inputs = {
            {{{32, 1, 10}, LOOP_IN_TYPE::INVARIANT}, {{32, 1, 10}, LOOP_IN_TYPE::INVARIANT}, {{32, 1, 10}, LOOP_IN_TYPE::MERGED}},
    };
    std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                             InferenceEngine::Precision::FP16};

    INSTANTIATE_TEST_SUITE_P(smoke_LoopCommonZeroClip, LoopTest,
                            ::testing::Combine(
                                    ::testing::ValuesIn(execute_first_iteration),
                                    ::testing::ValuesIn(is_body_condition_const),
                                    ::testing::ValuesIn(body_condition),
                                    ::testing::ValuesIn(trip_count),
                                    ::testing::ValuesIn(inputs),
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                            LoopTest::getTestCaseName);

    static const std::vector<std::tuple<bool, int64_t, int64_t, int64_t>> static_loop_types {
            //  GCC4.8 limitation: have to specify type of each element in list
            //                               static_trip_count |  max | dynamic_exit | axis
            std::tuple<bool, int64_t, int64_t, int64_t>{  true ,  5, -1, -1 },  // n_iter 5, no dynamic exit
            std::tuple<bool, int64_t, int64_t, int64_t>{  true ,  5,  3, -1 },  // n_iter 3, dynamic exit on 3
            std::tuple<bool, int64_t, int64_t, int64_t>{  true ,  5,  7, -1 },  // n_iter 5, dynamic exit not reached
            std::tuple<bool, int64_t, int64_t, int64_t>{  true , -1,  5, -1 },  // n_iter 5, inf loop with dynamic exit on 5
            std::tuple<bool, int64_t, int64_t, int64_t>{  true ,  5, -1,  1 },  // n_iter 5, const for loop with auto concatenated out
            std::tuple<bool, int64_t, int64_t, int64_t>{ false ,  5, -1, -1 },  // |
            std::tuple<bool, int64_t, int64_t, int64_t>{ false ,  5,  3, -1 },  // | same with dynamic trip count
            std::tuple<bool, int64_t, int64_t, int64_t>{ false ,  5,  7, -1 },  // |
            std::tuple<bool, int64_t, int64_t, int64_t>{ false , -1,  5, -1 }   // |
    };

    using namespace testing;
    using namespace InferenceEngine;

    INSTANTIATE_TEST_SUITE_P(smoke_StaticShapeLoop, StaticShapeLoopTest,
                            Combine(
                                    ValuesIn(std::vector<bool>{true, false}),
                                    Values(true),
                                    ValuesIn(static_loop_types),
                                    Values<int64_t>(7),
                                    Values<InferenceEngine::SizeVector>({2, 1, 4}),
                                    Values<InferenceEngine::Precision>(Precision::FP32, Precision::I32),
                                    Values(CommonTestUtils::DEVICE_CPU)));
    using namespace testing;
    INSTANTIATE_TEST_SUITE_P(smoke_TrivialLoop, TrivialLoopTest,
                            Combine(
                                    Values<InferenceEngine::Precision>(Precision::FP32, Precision::I32),
                                    Values<InferenceEngine::SizeVector>({2, 3, 4}),
                                    Values(CommonTestUtils::DEVICE_CPU)));

}  // namespace
