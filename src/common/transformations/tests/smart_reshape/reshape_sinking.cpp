// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset9.hpp"

struct ReshapeSinkingAttributes {
    ov::element::Type_t data_et;
    ov::PartialShape input_shape;
    ov::PartialShape new_shape;
    std::vector<int64_t> output_pattern;
    std::vector<int64_t> output_pattern_back;
    ov::Shape mm_second_input_shape;
    bool transpose_a, transpose_b;
};

class ReshapeSinkingTest : public testing::WithParamInterface<ReshapeSinkingAttributes>,
                           public ov::test::TestsCommon {};

TEST_P(ReshapeSinkingTest, ReshapeSinkingOnlyMatMul) {
    auto p = GetParam();

    std::shared_ptr<ov::Model> model(nullptr);
    {
        auto parameter = std::make_shared<ov::opset9::Parameter>(p.data_et, p.input_shape);
        auto reshape = std::make_shared<ov::opset9::Reshape>(parameter, create_constant(p.output_pattern), false);
        auto matmul =
            std::make_shared<ov::opset9::MatMul>(reshape,
                                                 ov::op::v0::Constant::create(p.data_et, p.mm_second_input_shape, {0}),
                                                 p.transpose_a,
                                                 p.transpose_b);
        auto reshape_back =
            std::make_shared<ov::opset9::Reshape>(matmul, create_constant(p.output_pattern_back), false);
        model = std::make_shared<ov::Model>(ov::NodeVector{reshape_back}, ov::ParameterVector{parameter});
    }
    OV_ASSERT_NO_THROW(model->reshape(p.new_shape));
}

class ReshapeSinkingTestWithAdd : public testing::WithParamInterface<ReshapeSinkingAttributes>,
                                  public ov::test::TestsCommon {};

TEST_P(ReshapeSinkingTestWithAdd, ReshapeSinkingMatMulAdd) {
    auto p = GetParam();

    std::shared_ptr<ov::Model> model(nullptr);
    {
        auto parameter = std::make_shared<ov::opset9::Parameter>(p.data_et, p.input_shape);
        auto reshape = std::make_shared<ov::opset9::Reshape>(parameter, create_constant(p.output_pattern), false);
        auto matmul =
            std::make_shared<ov::opset9::MatMul>(reshape,
                                                 ov::op::v0::Constant::create(p.data_et, p.mm_second_input_shape, {0}),
                                                 p.transpose_a,
                                                 p.transpose_b);
        auto add = std::make_shared<ov::opset9::Add>(matmul, ov::op::v0::Constant::create(p.data_et, {1, 37}, {0}));
        auto reshape_back = std::make_shared<ov::opset9::Reshape>(add, create_constant(p.output_pattern_back), false);
        model = std::make_shared<ov::Model>(ov::NodeVector{reshape_back}, ov::ParameterVector{parameter});
    }
    OV_ASSERT_NO_THROW(model->reshape(p.new_shape));
}

static std::vector<ReshapeSinkingAttributes> params = {
    ReshapeSinkingAttributes{ov::element::f32,
                             {10, 30, 512},
                             {20, 30, 512},
                             {-1, 512},
                             {10, 30, 37},
                             {37, 512},
                             false,
                             true},
    ReshapeSinkingAttributes{ov::element::f32,
                             {-1, 30, 512},
                             {20, 30, 512},
                             {-1, 512},
                             {10, 30, 37},
                             {37, 512},
                             false,
                             true},
    ReshapeSinkingAttributes{ov::element::f32,
                             {1, 3, 4, 512},
                             {2, 3, 4, 512},
                             {-1, 512},
                             {1, 3, 4, 37},
                             {37, 512},
                             false,
                             true},
    ReshapeSinkingAttributes{ov::element::f32,
                             {1, 3, 4, 512},
                             {2, 3, 4, 512},
                             {-1, 512},
                             {1, 3, 4, 37},
                             {512, 37},
                             false,
                             false},
};

INSTANTIATE_TEST_SUITE_P(SmartReshapeTests, ReshapeSinkingTest, ::testing::ValuesIn(params));
INSTANTIATE_TEST_SUITE_P(SmartReshapeTests, ReshapeSinkingTestWithAdd, ::testing::ValuesIn(params));
