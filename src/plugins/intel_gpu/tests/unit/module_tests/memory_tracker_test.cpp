// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include "intel_gpu/runtime/memory.hpp"

using namespace cldnn;
using namespace ::tests;

struct mem_tracker_test_params {
    std::vector<ov::Shape> in_shapes;
    ov::Shape expected_predicted_shape;
    float buffers_preallocation_ratio;
    bool can_reuse_buffer;
};

class memory_tracker_test : public testing::TestWithParam<mem_tracker_test_params> {};
TEST_P(memory_tracker_test, preallocation) {
    auto p = GetParam();
    auto& in_shapes = p.in_shapes;
    auto& expected_predicted_shape = p.expected_predicted_shape;
    auto& engine = get_test_engine();

    MemoryUsageTracker memory_tracker(&engine, p.buffers_preallocation_ratio);
    std::pair<bool, ov::Shape> result;

    for (auto& shape : in_shapes)
        result = memory_tracker.predict_preallocated_shape_size("dummy_name", shape, p.can_reuse_buffer);

    ASSERT_TRUE(result.first == !expected_predicted_shape.empty());
    ASSERT_EQ(result.second, expected_predicted_shape);
}

INSTANTIATE_TEST_SUITE_P(smoke, memory_tracker_test,
    testing::ValuesIn(std::vector<mem_tracker_test_params>{
        // Preallocation for next N iterations tests
        {{{1,1}, {1,1}, {1,1}}, {}, 1.0f, false},
        {{{1,1}, {1,21}, {1,31}}, {}, 1.0f, false},
        {{{1,3}, {1,2}, {1,1}}, {}, 1.0f, false},
        {{{1,1}, {1,2}, {1,3}}, {1,13}, 1.0f, false},
        {{{1,1}, {1,2}, {1,3}}, {1,13}, 1.1f, false},
        {{{1,1,1}, {1,2,2}, {1,3,3}}, {1,13,13}, 1.0f, false},
        {{{1,1,1}, {1,2,2}, {1,3,3}}, {1,13,13}, 1.1f, false},
        {{{1,1,1}, {1,3,2}, {1,7,3}}, {}, 1.0f, false},
        {{{1,1,1}, {1,1,3}, {1,1,5}}, {1,1,25}, 1.0f, false},
        {{{1,1,1}, {1,1,3}, {1,1,5}}, {1,1,25}, 1.1f, false},
        {{{1,1,1}, {1,1,3}, {1,1,5}}, {1,1,25}, 1.0f, false},
        {{{1,1,1}, {1,1,3}, {1,1,5}}, {1,1,25}, 1.1f, false},
        {{{1,1}, {1,1}, {1,1}, {1,1}, {1,1}, {1,1}}, {}, 1.0f, false},
        {{{1,10}, {1,1}, {1,2}, {1,3}}, {1,13}, 1.0f, false},
        {{{1,10}, {1,1}, {1,2}, {1,3}}, {1,13}, 1.1f, false},
        {{{1,1}, {1,1}, {1,1}}, {}, 1.0f, true},
        {{{1,1}, {1,2}, {1,3}}, {}, 1.0f, true},

        // Percentage preallocation tests
        {{{1,1}, {1,1}, {1,1}}, {1,1}, 1.1f, false},
        {{{1,1}, {1,128}, {1,256}}, {281, 1}, 1.1f, false},
        {{{1,3,128}, {1,3,112}, {1,3,418}, {1,3,512}}, {1689,1,1}, 1.1f, false},
        {{{1,1}, {1,1}, {1,1}}, {}, 1.1f, true},
        {{{1,1}, {1,128}, {1,256}}, {}, 1.1f, true},
        {{{1,3,128}, {1,3,112}, {1,3,418}, {1,3,512}}, {}, 1.1f, true},
    }));
