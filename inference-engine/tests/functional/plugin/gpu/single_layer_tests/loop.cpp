// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include "single_layer_tests/loop.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;
using namespace InferenceEngine;

namespace {
    static const std::vector<std::tuple<bool, int64_t, int64_t, int64_t>> static_loop_types {
            //  GCC4.8 limitation: have to specify type of each element in list
            //                               static_trip_count |  max | dynamic_exit | axis
            std::tuple<bool, int64_t, int64_t, int64_t>{  true ,  5, -1, -1 },  // n_iter 5, no dynamic exit
        //     std::tuple<bool, int64_t, int64_t, int64_t>{  true ,  5,  3, -1 },  // n_iter 3, dynamic exit on 3
        //     std::tuple<bool, int64_t, int64_t, int64_t>{  true ,  5,  7, -1 },  // n_iter 5, dynamic exit not reached
        //     std::tuple<bool, int64_t, int64_t, int64_t>{  true , -1,  5, -1 },  // n_iter 5, inf loop with dynamic exit on 5
            std::tuple<bool, int64_t, int64_t, int64_t>{  true ,  5, -1,  1 },  // n_iter 5, const for loop with auto concatenated out
        //     std::tuple<bool, int64_t, int64_t, int64_t>{ false ,  5, -1, -1 },  // |
        //     std::tuple<bool, int64_t, int64_t, int64_t>{ false ,  5,  3, -1 },  // | same with dynamic trip count
        //     std::tuple<bool, int64_t, int64_t, int64_t>{ false ,  5,  7, -1 },  // |
        //     std::tuple<bool, int64_t, int64_t, int64_t>{ false , -1,  5, -1 }   // |
    };

    INSTANTIATE_TEST_CASE_P(smoke_StaticShapeLoop, StaticShapeLoopTest,
                            testing::Combine(
                            /* unrolling */ testing::ValuesIn(std::vector<bool>{false}),
                            /* static_continue_cond */ testing::Values(true),
                            /* args_papck */ testing::ValuesIn(static_loop_types),
                            /* start_value */ testing::Values<int64_t>(0),
                            /* data_shape */ testing::Values<InferenceEngine::SizeVector>({2, 1, 4}),
                            /* data_prc */ testing::Values<InferenceEngine::Precision>(Precision::FP32, Precision::I32),
                            /* device */ testing::Values<std::string>(CommonTestUtils::DEVICE_GPU)));
//     INSTANTIATE_TEST_CASE_P(smoke_TrivialLoop, TrivialLoopTestGPU,
//                             testing::Combine(
//                                     testing::Values<InferenceEngine::Precision>(Precision::FP32, Precision::I32),
//                                     testing::Values<InferenceEngine::SizeVector>({2, 3, 4}),
//                                     testing::Values(CommonTestUtils::DEVICE_GPU)));

}  // namespace
