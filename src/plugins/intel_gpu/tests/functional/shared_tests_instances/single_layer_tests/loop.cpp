// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/loop.hpp"
#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::StaticShapeLoopLayerTest;

std::vector<ov::element::Type> model_types = {
    ov::element::f32,
    ov::element::i32
};

static const std::vector<std::tuple<bool, int64_t, int64_t, int64_t>> static_loop_types_axis_0 {
    //  GCC4.8 limitation: have to specify type of each element in list
    //                               static_trip_count |  max | dynamic_exit | axis
    std::tuple<bool, int64_t, int64_t, int64_t>{  true ,  10, -1, 0 },  // n_iter 10, no dynamic exit
};

std::vector<ov::Shape> inputs_0 = {
    {1, 4, 2}
};

INSTANTIATE_TEST_SUITE_P(smoke_StaticShapeLoop_axis_0, StaticShapeLoopLayerTest,
                        testing::Combine(
                        /* unrolling */ testing::ValuesIn(std::vector<bool>{false}),
                        /* static_continue_cond */ testing::Values(true),
                        /* args_papck */ testing::ValuesIn(static_loop_types_axis_0),
                        /* start_value */ testing::Values<int64_t>(0),
                        /* data_shape */ testing::ValuesIn(inputs_0),
                        /* data_prc */ testing::ValuesIn(model_types),
                        /* device */ testing::Values<std::string>(ov::test::utils::DEVICE_GPU)),
                        StaticShapeLoopLayerTest::getTestCaseName);

static const std::vector<std::tuple<bool, int64_t, int64_t, int64_t>> static_loop_types_1 {
    //  GCC4.8 limitation: have to specify type of each element in list
    //                               static_trip_count |  max | dynamic_exit | axis
    std::tuple<bool, int64_t, int64_t, int64_t>{  true ,  5, -1,  1 },  // n_iter 5, no dynamic exit
};

std::vector<ov::Shape> inputs_1 = {
    {2, 1, 4, 6}
};

INSTANTIATE_TEST_SUITE_P(smoke_StaticShapeLoop_axis_1, StaticShapeLoopLayerTest,
                        testing::Combine(
                        /* unrolling */ testing::ValuesIn(std::vector<bool>{false}),
                        /* static_continue_cond */ testing::Values(true),
                        /* args_papck */ testing::ValuesIn(static_loop_types_1),
                        /* start_value */ testing::Values<int64_t>(0),
                        /* data_shape */ testing::ValuesIn(inputs_1),
                        /* data_prc */ testing::ValuesIn(model_types),
                        /* device */ testing::Values<std::string>(ov::test::utils::DEVICE_GPU)),
                        StaticShapeLoopLayerTest::getTestCaseName);

static const std::vector<std::tuple<bool, int64_t, int64_t, int64_t>> static_loop_types_2 {
    //  GCC4.8 limitation: have to specify type of each element in list
    //                               static_trip_count |  max | dynamic_exit | axis
    std::tuple<bool, int64_t, int64_t, int64_t>{  true ,  10, -1,  2 },  // n_iter 10, no dynamic exit
};

std::vector<ov::Shape> inputs_2 = {
    {2, 4, 1, 6}
};

INSTANTIATE_TEST_SUITE_P(smoke_StaticShapeLoop_axis_2, StaticShapeLoopLayerTest,
                        testing::Combine(
                        /* unrolling */ testing::ValuesIn(std::vector<bool>{false}),
                        /* static_continue_cond */ testing::Values(true),
                        /* args_papck */ testing::ValuesIn(static_loop_types_2),
                        /* start_value */ testing::Values<int64_t>(0),
                        /* data_shape */ testing::ValuesIn(inputs_2),
                        /* data_prc */ testing::ValuesIn(model_types),
                        /* device */ testing::Values<std::string>(ov::test::utils::DEVICE_GPU)),
                        StaticShapeLoopLayerTest::getTestCaseName);

static const std::vector<std::tuple<bool, int64_t, int64_t, int64_t>> static_loop_types_no_auto_concat {
    //  GCC4.8 limitation: have to specify type of each element in list
    //                               static_trip_count |  max | dynamic_exit | axis
    std::tuple<bool, int64_t, int64_t, int64_t>{  true ,  10, -1, -1 },  // n_iter 5, no dynamic exit
};

std::vector<ov::Shape> inputs_no_auto_concat = {
    {4, 20, 12}
};

INSTANTIATE_TEST_SUITE_P(smoke_StaticShapeLoop_no_auto_concat, StaticShapeLoopLayerTest,
                        testing::Combine(
                        /* unrolling */ testing::ValuesIn(std::vector<bool>{false}),
                        /* static_continue_cond */ testing::Values(true),
                        /* args_papck */ testing::ValuesIn(static_loop_types_no_auto_concat),
                        /* start_value */ testing::Values<int64_t>(0),
                        /* data_shape */ testing::ValuesIn(inputs_no_auto_concat),
                        /* data_prc */ testing::ValuesIn(model_types),
                        /* device */ testing::Values<std::string>(ov::test::utils::DEVICE_GPU)),
                        StaticShapeLoopLayerTest::getTestCaseName);

static const std::vector<std::tuple<bool, int64_t, int64_t, int64_t>> static_loop_types_dynamic_exit {
    //  GCC4.8 limitation: have to specify type of each element in list
    //                               static_trip_count |  max | dynamic_exit | axis
    std::tuple<bool, int64_t, int64_t, int64_t>{  true ,  5,  3,  -1 },  // n_iter 3, dynamic exit on 3
    std::tuple<bool, int64_t, int64_t, int64_t>{  true ,  5,  7,   1 },  // n_iter 5, dynamic exit not reached
    std::tuple<bool, int64_t, int64_t, int64_t>{  true , -1,  5,  -1 },  // n_iter 5, inf loop with dynamic exit on 5
    std::tuple<bool, int64_t, int64_t, int64_t>{ false ,  5,  3,  -1 },  // | same with dynamic trip count
    std::tuple<bool, int64_t, int64_t, int64_t>{ false ,  5,  7,   1 },  // |
    std::tuple<bool, int64_t, int64_t, int64_t>{ false , -1,  5,  -1 }   // |
};

std::vector<ov::Shape> inputs_dynamic_exit = {
    {4, 1, 2}
};

INSTANTIATE_TEST_SUITE_P(smoke_StaticShapeLoop_dynamic_exit, StaticShapeLoopLayerTest,
                        testing::Combine(
                        /* unrolling */ testing::ValuesIn(std::vector<bool>{false}),
                        /* static_continue_cond */ testing::Values(true),
                        /* args_papck */ testing::ValuesIn(static_loop_types_dynamic_exit),
                        /* start_value */ testing::Values<int64_t>(0),
                        /* data_shape */ testing::ValuesIn(inputs_dynamic_exit),
                        /* data_prc */ testing::ValuesIn(model_types),
                        /* device */ testing::Values<std::string>(ov::test::utils::DEVICE_GPU)),
                        StaticShapeLoopLayerTest::getTestCaseName);
}  // namespace
