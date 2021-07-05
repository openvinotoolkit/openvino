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
    std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::I32
    };

    std::map<std::string, std::string> netConfigurations = {
        {GPUConfigParams::KEY_GPU_ENABLE_LOOP_UNROLLING, PluginConfigParams::NO}
    };

    static const std::vector<std::tuple<bool, int64_t, int64_t, int64_t>> static_loop_types_axis_0 {
        //  GCC4.8 limitation: have to specify type of each element in list
        //                               static_trip_count |  max | dynamic_exit | axis
        std::tuple<bool, int64_t, int64_t, int64_t>{  true ,  10, -1, 0 }  // n_iter 10, no dynamic exit
    };

    std::vector<InferenceEngine::SizeVector> inputs_0 = {
        {1, 4, 2}
    };

    INSTANTIATE_TEST_CASE_P(smoke_StaticShapeLoop_axis_0, StaticShapeLoopTest,
                            testing::Combine(
                            /* unrolling */ testing::ValuesIn(std::vector<bool>{false}),
                            /* static_continue_cond */ testing::Values(true),
                            /* args_papck */ testing::ValuesIn(static_loop_types_axis_0),
                            /* start_value */ testing::Values<int64_t>(0),
                            /* data_shape */ testing::ValuesIn(inputs_0),
                            /* data_prc */ testing::ValuesIn(netPrecisions),
                            /* device */ testing::Values<std::string>(CommonTestUtils::DEVICE_GPU),
                            /* configuration */ testing::Values<std::map<std::string, std::string>>(netConfigurations)));

    static const std::vector<std::tuple<bool, int64_t, int64_t, int64_t>> static_loop_types_1 {
        //  GCC4.8 limitation: have to specify type of each element in list
        //                               static_trip_count |  max | dynamic_exit | axis
        std::tuple<bool, int64_t, int64_t, int64_t>{  true ,  5, -1,  1 }  // n_iter 5, no dynamic exit
    };

    std::vector<InferenceEngine::SizeVector> inputs_1 = {
        {2, 1, 4, 6}
    };

    INSTANTIATE_TEST_CASE_P(smoke_StaticShapeLoop_axis_1, StaticShapeLoopTest,
                            testing::Combine(
                            /* unrolling */ testing::ValuesIn(std::vector<bool>{false}),
                            /* static_continue_cond */ testing::Values(true),
                            /* args_papck */ testing::ValuesIn(static_loop_types_1),
                            /* start_value */ testing::Values<int64_t>(0),
                            /* data_shape */ testing::ValuesIn(inputs_1),
                            /* data_prc */ testing::ValuesIn(netPrecisions),
                            /* device */ testing::Values<std::string>(CommonTestUtils::DEVICE_GPU),
                            /* configuration */ testing::Values<std::map<std::string, std::string>>(netConfigurations)));

    static const std::vector<std::tuple<bool, int64_t, int64_t, int64_t>> static_loop_types_2 {
        //  GCC4.8 limitation: have to specify type of each element in list
        //                               static_trip_count |  max | dynamic_exit | axis
        std::tuple<bool, int64_t, int64_t, int64_t>{  true ,  10, -1,  2 }  // n_iter 10, no dynamic exit
    };

    std::vector<InferenceEngine::SizeVector> inputs_2 = {
        {2, 4, 1, 6}
    };

    INSTANTIATE_TEST_CASE_P(smoke_StaticShapeLoop_axis_2, StaticShapeLoopTest,
                            testing::Combine(
                            /* unrolling */ testing::ValuesIn(std::vector<bool>{false}),
                            /* static_continue_cond */ testing::Values(true),
                            /* args_papck */ testing::ValuesIn(static_loop_types_2),
                            /* start_value */ testing::Values<int64_t>(0),
                            /* data_shape */ testing::ValuesIn(inputs_2),
                            /* data_prc */ testing::ValuesIn(netPrecisions),
                            /* device */ testing::Values<std::string>(CommonTestUtils::DEVICE_GPU),
                            /* configuration */ testing::Values<std::map<std::string, std::string>>(netConfigurations)));


    static const std::vector<std::tuple<bool, int64_t, int64_t, int64_t>> static_loop_types_no_auto_concat {
        //  GCC4.8 limitation: have to specify type of each element in list
        //                               static_trip_count |  max | dynamic_exit | axis
        std::tuple<bool, int64_t, int64_t, int64_t>{  true ,  10, -1, -1 }  // n_iter 5, no dynamic exit
    };

    std::vector<InferenceEngine::SizeVector> inputs_no_auto_concat = {
        {4, 20, 12}
    };

    INSTANTIATE_TEST_CASE_P(smoke_StaticShapeLoop_no_auto_concat, StaticShapeLoopTest,
                            testing::Combine(
                            /* unrolling */ testing::ValuesIn(std::vector<bool>{false}),
                            /* static_continue_cond */ testing::Values(true),
                            /* args_papck */ testing::ValuesIn(static_loop_types_no_auto_concat),
                            /* start_value */ testing::Values<int64_t>(0),
                            /* data_shape */ testing::ValuesIn(inputs_no_auto_concat),
                            /* data_prc */ testing::ValuesIn(netPrecisions),
                            /* device */ testing::Values<std::string>(CommonTestUtils::DEVICE_GPU),
                            /* configuration */ testing::Values<std::map<std::string, std::string>>(netConfigurations)));

}  // namespace
