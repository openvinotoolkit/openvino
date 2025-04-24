// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/loop.hpp"
#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::LoopLayerTest;
using ov::test::LOOP_IN_TYPE;

    // without clip values increase rapidly, so use only seq_lengths = 2
    std::vector<bool> execute_first_iteration{true};
    std::vector<bool> is_body_condition_const{true/*, false*/};
    std::vector<bool> body_condition{true/*, false*/}; // works only if is_body_condition_const == true
    std::vector<int64_t> trip_count{1, 10/*, -1*/}; // -1 means infinity
    std::vector<ov::Shape> input_shapes_static = {{32, 1, 10}};

    std::vector<std::vector<LOOP_IN_TYPE>> inputs_types = {
        {LOOP_IN_TYPE::INVARIANT},
        {LOOP_IN_TYPE::MERGED}};

    std::vector<ov::element::Type> model_types = {
        ov::element::f32,
        ov::element::f16};

    INSTANTIATE_TEST_SUITE_P(smoke_LoopCommonZeroClip, LoopLayerTest,
                            ::testing::Combine(
                                    ::testing::ValuesIn(execute_first_iteration),
                                    ::testing::ValuesIn(is_body_condition_const),
                                    ::testing::ValuesIn(body_condition),
                                    ::testing::ValuesIn(trip_count),
                                    ::testing::Values(ov::test::static_shapes_to_test_representation(input_shapes_static)),
                                    ::testing::ValuesIn(inputs_types),
                                    ::testing::ValuesIn(model_types),
                                    ::testing::Values(ov::test::utils::DEVICE_CPU)),
                            LoopLayerTest::getTestCaseName);
}  // namespace
