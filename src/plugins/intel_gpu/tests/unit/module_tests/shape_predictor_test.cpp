// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include "intel_gpu/runtime/shape_predictor.hpp"

using namespace cldnn;
using namespace ::tests;

struct shape_predictor_test_params {
    std::vector<ov::Shape> in_shapes;
    ov::Shape expected_predicted_shape;
    float buffers_preallocation_ratio;
    bool can_reuse_buffer;
};

class shape_predictor_tests : public testing::TestWithParam<shape_predictor_test_params> {};
TEST_P(shape_predictor_tests, prediction) {
    auto p = GetParam();
    auto& in_shapes = p.in_shapes;
    auto& expected_predicted_shape = p.expected_predicted_shape;
    auto& engine = get_test_engine();

    ShapePredictor sp(&engine, p.buffers_preallocation_ratio);
    std::pair<bool, ov::Shape> result;

    for (auto& shape : in_shapes)
        result = sp.predict_preallocation_shape("dummy_name", cldnn::layout(shape,
                                                                            ov::element::f32,
                                                                            cldnn::format::get_default_format(shape.size())),
                                                p.can_reuse_buffer);

    ASSERT_TRUE(result.first == !expected_predicted_shape.empty());
    ASSERT_EQ(result.second, expected_predicted_shape);
}

INSTANTIATE_TEST_SUITE_P(smoke, shape_predictor_tests,
    testing::ValuesIn(std::vector<shape_predictor_test_params>{
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
        {{{1,3,480,720}, {3,3,480,720}, {5,3,480,720}}, {}, 1.0f, false},
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

class shape_predictor_tests_b_fs_yx_fsv16 : public testing::TestWithParam<shape_predictor_test_params> {};
TEST_P(shape_predictor_tests_b_fs_yx_fsv16, prediction) {
    auto p = GetParam();
    auto& in_shapes = p.in_shapes;
    auto& expected_predicted_shape = p.expected_predicted_shape;
    auto& engine = get_test_engine();

    ShapePredictor sp(&engine, p.buffers_preallocation_ratio);
    std::pair<bool, ov::Shape> result;

    for (auto& shape : in_shapes)
        result = sp.predict_preallocation_shape("dummy_name", cldnn::layout(shape,
                                                                            ov::element::f32,
                                                                            cldnn::format::b_fs_yx_fsv16),
                                                p.can_reuse_buffer);

    ASSERT_TRUE(result.first == !expected_predicted_shape.empty());
    ASSERT_EQ(result.second, expected_predicted_shape);
}

INSTANTIATE_TEST_SUITE_P(smoke, shape_predictor_tests_b_fs_yx_fsv16,
    testing::ValuesIn(std::vector<shape_predictor_test_params>{
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
        {{{1,3,480,720}, {3,3,480,720}, {5,3,480,720}}, {}, 1.0f, false},
        {{{1,1}, {1,1}, {1,1}}, {}, 1.0f, true},
        {{{1,1}, {1,2}, {1,3}}, {}, 1.0f, true},

        // Percentage preallocation tests
        {{{1,1}, {1,1}, {1,1}}, {}, 1.1f, false},
        {{{1,1}, {1,128}, {1,256}}, {}, 1.1f, false},
        {{{1,3,128}, {1,3,112}, {1,3,418}, {1,3,512}}, {}, 1.1f, false},
        {{{1,1}, {1,1}, {1,1}}, {}, 1.1f, true},
        {{{1,1}, {1,128}, {1,256}}, {}, 1.1f, true},
        {{{1,3,128}, {1,3,112}, {1,3,418}, {1,3,512}}, {}, 1.1f, true},
    }));
