// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>
#include <queue>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <transformations/smart_reshape/broadcast_const_range_replacement.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"


using namespace testing;
using namespace ngraph;

TEST_F(TransformationTestsF, BroadcastConstRangeReplacement_dim_match) {
    {
        constexpr auto elem_count = 236;
        constexpr auto data_elem_type = element::i32;
        constexpr auto target_shape_elem_type = element::i64;

        std::vector<int32_t> sequence_pattern(elem_count);
        std::iota(sequence_pattern.begin(), sequence_pattern.end(), 0);
        auto data_to_broadcast = ngraph::opset8::Constant::create(data_elem_type, {1, elem_count}, sequence_pattern);
        auto target_shape = ngraph::opset8::Constant::create(target_shape_elem_type, {4}, {2, 3, 4, elem_count});
        auto broadcast_node = std::make_shared<ngraph::opset8::Broadcast>(data_to_broadcast, target_shape, ngraph::op::BroadcastType::BIDIRECTIONAL);

        function = std::make_shared<Function>(OutputVector{broadcast_node}, ParameterVector{});

        manager.register_pass<pass::BroadcastConstRangeReplacement>();
    }
    {
        constexpr auto elem_count = 236;
        constexpr auto data_elem_type = element::i32;
        constexpr auto target_shape_elem_type = element::i64;

        auto target_shape = ngraph::opset8::Constant::create(element::i64, {4}, {2, 3, 4, elem_count});

        const auto target_dim_neg_index = -1;
        const auto axis_node = ngraph::opset8::Constant::create(ngraph::element::i32, Shape{}, {0});
        const auto target_dim_index_node = ngraph::opset8::Constant::create(ngraph::element::i64, Shape{}, {target_dim_neg_index});
        const auto gather_dim = std::make_shared<ngraph::opset8::Gather>(target_shape, target_dim_index_node, axis_node);

        const auto one_dim_const = ngraph::opset8::Constant::create(target_shape_elem_type, {}, {1});
        const auto dim_check_one = std::make_shared<ngraph::opset8::Equal>(gather_dim, one_dim_const);

        const auto start = ngraph::opset8::Constant::create(data_elem_type, {}, {0});
        const auto original_end = ngraph::opset8::Constant::create(data_elem_type, {}, {elem_count});

        const auto cast_gather_dim = std::make_shared<ngraph::opset8::Convert>(gather_dim, data_elem_type);
        const auto select_end = std::make_shared<ngraph::opset8::Select>(dim_check_one, original_end, cast_gather_dim);

        const auto default_range_step = ngraph::opset8::Constant::create(data_elem_type, {}, {1});
        const auto range = std::make_shared<ngraph::opset8::Range>(start, select_end, default_range_step, data_elem_type);
        const auto axes_to_unsqueeze = ngraph::opset8::Constant::create(ngraph::element::i64, Shape{1}, {0});
        const auto unsqueeze_range = std::make_shared<ngraph::opset8::Unsqueeze>(range, axes_to_unsqueeze);

        const auto broadcast_node = std::make_shared<ngraph::opset8::Broadcast>(unsqueeze_range, target_shape, ngraph::op::BroadcastType::BIDIRECTIONAL);

        function_ref = std::make_shared<Function>(OutputVector{broadcast_node}, ParameterVector{});
    }
}

TEST_F(TransformationTestsF, BroadcastConstRangeReplacement_dim_one) {
    {
        constexpr auto elem_count = 236;
        constexpr auto data_elem_type = element::i32;
        constexpr auto target_shape_elem_type = element::i64;

        std::vector<int32_t> sequence_pattern(elem_count);
        std::iota(sequence_pattern.begin(), sequence_pattern.end(), 0);
        auto data_to_broadcast = ngraph::opset8::Constant::create(data_elem_type, {1, elem_count}, sequence_pattern);
        auto target_shape = ngraph::opset8::Constant::create(target_shape_elem_type, {4}, {2, 3, 4, 1});
        auto broadcast_node = std::make_shared<ngraph::opset8::Broadcast>(data_to_broadcast, target_shape, ngraph::op::BroadcastType::BIDIRECTIONAL);

        function = std::make_shared<Function>(OutputVector{broadcast_node}, ParameterVector{});

        manager.register_pass<pass::BroadcastConstRangeReplacement>();
    }
    {
        constexpr auto elem_count = 236;
        constexpr auto data_elem_type = element::i32;
        constexpr auto target_shape_elem_type = element::i64;

        auto target_shape = ngraph::opset8::Constant::create(element::i64, {4}, {2, 3, 4, 1});

        const auto target_dim_neg_index = -1;
        const auto axis_node = ngraph::opset8::Constant::create(ngraph::element::i32, Shape{}, {0});
        const auto target_dim_index_node = ngraph::opset8::Constant::create(ngraph::element::i64, Shape{}, {target_dim_neg_index});
        const auto gather_dim = std::make_shared<ngraph::opset8::Gather>(target_shape, target_dim_index_node, axis_node);

        // If the corresponding target dim is 1, use the original end of range
        const auto one_dim_const = ngraph::opset8::Constant::create(target_shape_elem_type, {}, {1});
        const auto dim_check_one = std::make_shared<ngraph::opset8::Equal>(gather_dim, one_dim_const);

        const auto start = ngraph::opset8::Constant::create(data_elem_type, {}, {0});
        const auto original_end = ngraph::opset8::Constant::create(data_elem_type, {}, {elem_count});

        const auto cast_gather_dim = std::make_shared<ngraph::opset8::Convert>(gather_dim, data_elem_type);
        const auto select_end = std::make_shared<ngraph::opset8::Select>(dim_check_one, original_end, cast_gather_dim);

        const auto default_range_step = ngraph::opset8::Constant::create(data_elem_type, {}, {1});
        const auto range = std::make_shared<ngraph::opset8::Range>(start, select_end, default_range_step, data_elem_type);
        const auto axes_to_unsqueeze = ngraph::opset8::Constant::create(ngraph::element::i64, Shape{1}, {0});
        const auto unsqueeze_range = std::make_shared<ngraph::opset8::Unsqueeze>(range, axes_to_unsqueeze);

        const auto broadcast_node = std::make_shared<ngraph::opset8::Broadcast>(unsqueeze_range, target_shape, ngraph::op::BroadcastType::BIDIRECTIONAL);

        function_ref = std::make_shared<Function>(OutputVector{broadcast_node}, ParameterVector{});
    }
}

TEST_F(TransformationTestsF, BroadcastConstRangeReplacement_target_shapeof) {
    {
        constexpr auto elem_count = 236;
        constexpr auto data_elem_type = element::i32;
        constexpr auto target_shape_elem_type = element::i64;

        std::vector<int32_t> sequence_pattern(elem_count);
        std::iota(sequence_pattern.begin(), sequence_pattern.end(), 0);

        auto data_param = std::make_shared<ngraph::opset8::Parameter>(data_elem_type, Shape{2, 3, 4, elem_count});
        auto target_shape = std::make_shared<ngraph::opset8::ShapeOf>(data_param);

        auto data_to_broadcast = ngraph::opset8::Constant::create(data_elem_type, {1, elem_count}, sequence_pattern);
        auto broadcast_node = std::make_shared<ngraph::opset8::Broadcast>(data_to_broadcast, target_shape, ngraph::op::BroadcastType::BIDIRECTIONAL);

        function = std::make_shared<Function>(OutputVector{broadcast_node}, ParameterVector{data_param});

        manager.register_pass<pass::BroadcastConstRangeReplacement>();
    }
    {
        constexpr auto elem_count = 236;
        constexpr auto data_elem_type = element::i32;
        constexpr auto target_shape_elem_type = element::i64;

        auto data_param = std::make_shared<ngraph::opset8::Parameter>(data_elem_type, Shape{2, 3, 4, elem_count});
        auto target_shape = std::make_shared<ngraph::opset8::ShapeOf>(data_param);

        const auto target_dim_neg_index = -1;
        const auto axis_node = ngraph::opset8::Constant::create(ngraph::element::i32, Shape{}, {0});
        const auto target_dim_index_node = ngraph::opset8::Constant::create(ngraph::element::i64, Shape{}, {target_dim_neg_index});
        const auto gather_dim = std::make_shared<ngraph::opset8::Gather>(target_shape, target_dim_index_node, axis_node);

        // If the corresponding target dim is 1, use the original end of range
        const auto one_dim_const = ngraph::opset8::Constant::create(target_shape_elem_type, {}, {1});
        const auto dim_check_one = std::make_shared<ngraph::opset8::Equal>(gather_dim, one_dim_const);

        const auto start = ngraph::opset8::Constant::create(data_elem_type, {}, {0});
        const auto original_end = ngraph::opset8::Constant::create(data_elem_type, {}, {elem_count});

        const auto cast_gather_dim = std::make_shared<ngraph::opset8::Convert>(gather_dim, data_elem_type);
        const auto select_end = std::make_shared<ngraph::opset8::Select>(dim_check_one, original_end, cast_gather_dim);

        const auto default_range_step = ngraph::opset8::Constant::create(data_elem_type, {}, {1});
        const auto range = std::make_shared<ngraph::opset8::Range>(start, select_end, default_range_step, data_elem_type);
        const auto axes_to_unsqueeze = ngraph::opset8::Constant::create(ngraph::element::i64, Shape{1}, {0});
        const auto unsqueeze_range = std::make_shared<ngraph::opset8::Unsqueeze>(range, axes_to_unsqueeze);

        const auto broadcast_node = std::make_shared<ngraph::opset8::Broadcast>(unsqueeze_range, target_shape, ngraph::op::BroadcastType::BIDIRECTIONAL);

        function_ref = std::make_shared<Function>(OutputVector{broadcast_node}, ParameterVector{data_param});
    }
}

TEST_F(TransformationTestsF, BroadcastConstRangeReplacement_target_shapeof_mixed_dims) {
    {
        constexpr auto elem_count = 236;
        constexpr auto data_elem_type = element::i32;
        constexpr auto target_shape_elem_type = element::i64;

        std::vector<int32_t> sequence_pattern(elem_count);
        std::iota(sequence_pattern.begin(), sequence_pattern.end(), 0);

        auto data_param = std::make_shared<ngraph::opset8::Parameter>(data_elem_type, Shape{2, 3, elem_count, 4});
        auto target_shape = std::make_shared<ngraph::opset8::ShapeOf>(data_param);

        auto data_to_broadcast = ngraph::opset8::Constant::create(data_elem_type, {1, 1, elem_count, 1}, sequence_pattern);
        auto broadcast_node = std::make_shared<ngraph::opset8::Broadcast>(data_to_broadcast, target_shape, ngraph::op::BroadcastType::BIDIRECTIONAL);

        function = std::make_shared<Function>(OutputVector{broadcast_node}, ParameterVector{data_param});

        manager.register_pass<pass::BroadcastConstRangeReplacement>();
    }
    {
        constexpr auto elem_count = 236;
        constexpr auto data_elem_type = element::i32;
        constexpr auto target_shape_elem_type = element::i64;

        auto data_param = std::make_shared<ngraph::opset8::Parameter>(data_elem_type, Shape{2, 3, elem_count, 4});
        auto target_shape = std::make_shared<ngraph::opset8::ShapeOf>(data_param);

        const auto target_dim_neg_index = -2;
        const auto axis_node = ngraph::opset8::Constant::create(ngraph::element::i32, Shape{}, {0});
        const auto target_dim_index_node = ngraph::opset8::Constant::create(ngraph::element::i64, Shape{}, {target_dim_neg_index});
        const auto gather_dim = std::make_shared<ngraph::opset8::Gather>(target_shape, target_dim_index_node, axis_node);

        const auto one_dim_const = ngraph::opset8::Constant::create(target_shape_elem_type, {}, {1});
        const auto dim_check_one = std::make_shared<ngraph::opset8::Equal>(gather_dim, one_dim_const);

        const auto start = ngraph::opset8::Constant::create(data_elem_type, {}, {0});
        const auto original_end = ngraph::opset8::Constant::create(data_elem_type, {}, {elem_count});

        const auto cast_gather_dim = std::make_shared<ngraph::opset8::Convert>(gather_dim, data_elem_type);
        const auto select_end = std::make_shared<ngraph::opset8::Select>(dim_check_one, original_end, cast_gather_dim);

        const auto default_range_step = ngraph::opset8::Constant::create(data_elem_type, {}, {1});
        const auto range = std::make_shared<ngraph::opset8::Range>(start, select_end, default_range_step, data_elem_type);

        // Axes to unsqueeze without target dim index
        const auto axes_to_unsqueeze = ngraph::opset8::Constant::create(ngraph::element::i64, Shape{3}, {0, 1, 3});
        const auto unsqueeze_range = std::make_shared<ngraph::opset8::Unsqueeze>(range, axes_to_unsqueeze);

        const auto broadcast_node = std::make_shared<ngraph::opset8::Broadcast>(unsqueeze_range, target_shape, ngraph::op::BroadcastType::BIDIRECTIONAL);

        function_ref = std::make_shared<Function>(OutputVector{broadcast_node}, ParameterVector{data_param});
    }
}

TEST_F(TransformationTestsF, BroadcastConstRangeReplacementNeg_other_mode) {
    {
        constexpr auto elem_count = 236;
        constexpr auto data_elem_type = element::i32;
        constexpr auto target_shape_elem_type = element::i64;

        std::vector<int32_t> sequence_pattern(elem_count);
        std::iota(sequence_pattern.begin(), sequence_pattern.end(), 0);
        auto data_to_broadcast = ngraph::opset8::Constant::create(data_elem_type, {1, elem_count}, sequence_pattern);
        auto target_shape = ngraph::opset8::Constant::create(target_shape_elem_type, {4}, {2, 3, 4, elem_count});
        auto broadcast_node = std::make_shared<ngraph::opset8::Broadcast>(data_to_broadcast, target_shape, ngraph::op::BroadcastType::NUMPY);

        function = std::make_shared<Function>(OutputVector{broadcast_node}, ParameterVector{});

        manager.register_pass<pass::BroadcastConstRangeReplacement>();
    }
}

TEST_F(TransformationTestsF, BroadcastConstRangeReplacementNeg_reversed_sequence) {
    {
        constexpr auto elem_count = 236;
        constexpr auto data_elem_type = element::i32;
        constexpr auto target_shape_elem_type = element::i64;

        std::vector<int32_t> sequence_pattern(elem_count);
        std::iota(sequence_pattern.rbegin(), sequence_pattern.rend(), 0);
        auto data_to_broadcast = ngraph::opset8::Constant::create(data_elem_type, {1, elem_count}, sequence_pattern);
        auto target_shape = ngraph::opset8::Constant::create(target_shape_elem_type, {4}, {2, 3, 4, elem_count});
        auto broadcast_node = std::make_shared<ngraph::opset8::Broadcast>(data_to_broadcast, target_shape, ngraph::op::BroadcastType::BIDIRECTIONAL);

        function = std::make_shared<Function>(OutputVector{broadcast_node}, ParameterVector{});

        manager.register_pass<pass::BroadcastConstRangeReplacement>();
    }
}

TEST_F(TransformationTestsF, BroadcastConstRangeReplacementNeg_too_small) {
    {
        constexpr auto elem_count = 4;
        constexpr auto data_elem_type = element::i32;
        constexpr auto target_shape_elem_type = element::i64;

        std::vector<int32_t> sequence_pattern(elem_count);
        std::iota(sequence_pattern.begin(), sequence_pattern.end(), 0);
        auto data_to_broadcast = ngraph::opset8::Constant::create(data_elem_type, {1, elem_count}, sequence_pattern);
        auto target_shape = ngraph::opset8::Constant::create(target_shape_elem_type, {4}, {2, 3, 4, elem_count});
        auto broadcast_node = std::make_shared<ngraph::opset8::Broadcast>(data_to_broadcast, target_shape, ngraph::op::BroadcastType::NUMPY);

        function = std::make_shared<Function>(OutputVector{broadcast_node}, ParameterVector{});

        manager.register_pass<pass::BroadcastConstRangeReplacement>();
    }
}

TEST_F(TransformationTestsF, BroadcastConstRangeReplacementNeg_too_big) {
    {
        constexpr auto elem_count = 1024;
        constexpr auto data_elem_type = element::i32;
        constexpr auto target_shape_elem_type = element::i64;

        std::vector<int32_t> sequence_pattern(elem_count);
        std::iota(sequence_pattern.begin(), sequence_pattern.end(), 0);
        auto data_to_broadcast = ngraph::opset8::Constant::create(data_elem_type, {1, elem_count}, sequence_pattern);
        auto target_shape = ngraph::opset8::Constant::create(target_shape_elem_type, {4}, {2, 3, 4, elem_count});
        auto broadcast_node = std::make_shared<ngraph::opset8::Broadcast>(data_to_broadcast, target_shape, ngraph::op::BroadcastType::NUMPY);

        function = std::make_shared<Function>(OutputVector{broadcast_node}, ParameterVector{});

        manager.register_pass<pass::BroadcastConstRangeReplacement>();
    }
}

// Model reshape call test
TEST(SmartReshapeTests, BroadcastConstRangeReplacement_reshape) {
    {
        constexpr auto elem_count = 236;
        constexpr auto data_elem_type = element::i32;
        constexpr auto target_shape_elem_type = element::i64;

        std::vector<int32_t> sequence_pattern(elem_count);
        std::iota(sequence_pattern.begin(), sequence_pattern.end(), 0);

        auto data_param = std::make_shared<ngraph::opset8::Parameter>(data_elem_type, Shape{2, 3, 4, elem_count});
        auto target_shape = std::make_shared<ngraph::opset8::ShapeOf>(data_param);

        auto data_to_broadcast = ngraph::opset8::Constant::create(data_elem_type, {1, elem_count}, sequence_pattern);
        auto broadcast_node = std::make_shared<ngraph::opset8::Broadcast>(data_to_broadcast, target_shape, ngraph::op::BroadcastType::BIDIRECTIONAL);

        auto function = std::make_shared<Function>(OutputVector{broadcast_node}, ParameterVector{data_param});

        // BroadcastConstRangeReplacement is called as a part of SmartReshape
        EXPECT_NO_THROW(function->reshape(PartialShape{1, 189}));
    }
}
