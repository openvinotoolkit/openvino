// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/smart_reshape/broadcast_const_range_replacement.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <queue>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset8.hpp"

using namespace testing;
using namespace ov;

TEST_F(TransformationTestsF, BroadcastConstRangeReplacement_dim_match) {
    {
        constexpr auto elem_count = 236;
        constexpr auto data_elem_type = element::i32;
        constexpr auto target_shape_elem_type = element::i64;

        std::vector<int32_t> sequence_pattern(elem_count);
        std::iota(sequence_pattern.begin(), sequence_pattern.end(), 0);
        auto data_to_broadcast = opset8::Constant::create(data_elem_type, {1, elem_count}, sequence_pattern);
        auto target_shape = opset8::Constant::create(target_shape_elem_type, {4}, {2, 3, 4, elem_count});
        auto broadcast_node =
            std::make_shared<opset8::Broadcast>(data_to_broadcast, target_shape, op::BroadcastType::BIDIRECTIONAL);

        model = std::make_shared<Model>(OutputVector{broadcast_node}, ParameterVector{});

        manager.register_pass<ov::pass::BroadcastConstRangeReplacement>();
    }
    {
        constexpr auto elem_count = 236;
        constexpr auto data_elem_type = element::i32;
        constexpr auto target_shape_elem_type = element::i64;

        auto target_shape = opset8::Constant::create(element::i64, {4}, {2, 3, 4, elem_count});

        const auto target_dim_neg_index = -1;
        const auto axis_node = opset8::Constant::create(element::i32, Shape{}, {0});
        const auto target_dim_index_node = opset8::Constant::create(element::i64, Shape{}, {target_dim_neg_index});
        const auto gather_dim = std::make_shared<opset8::Gather>(target_shape, target_dim_index_node, axis_node);

        const auto one_dim_const = opset8::Constant::create(target_shape_elem_type, {}, {1});
        const auto dim_check_one = std::make_shared<opset8::Equal>(gather_dim, one_dim_const);

        const auto start = opset8::Constant::create(data_elem_type, {}, {0});
        const auto original_end = opset8::Constant::create(data_elem_type, {}, {elem_count});

        const auto cast_gather_dim = std::make_shared<opset8::Convert>(gather_dim, data_elem_type);
        const auto select_end = std::make_shared<opset8::Select>(dim_check_one, original_end, cast_gather_dim);

        const auto default_range_step = opset8::Constant::create(data_elem_type, {}, {1});
        const auto range = std::make_shared<opset8::Range>(start, select_end, default_range_step, data_elem_type);
        const auto axes_to_unsqueeze = opset8::Constant::create(element::i64, Shape{1}, {0});
        const auto unsqueeze_range = std::make_shared<opset8::Unsqueeze>(range, axes_to_unsqueeze);

        const auto broadcast_node =
            std::make_shared<opset8::Broadcast>(unsqueeze_range, target_shape, op::BroadcastType::BIDIRECTIONAL);

        model_ref = std::make_shared<Model>(OutputVector{broadcast_node}, ParameterVector{});
    }
}

TEST_F(TransformationTestsF, BroadcastConstRangeReplacement_dim_one) {
    {
        constexpr auto elem_count = 236;
        constexpr auto data_elem_type = element::i32;
        constexpr auto target_shape_elem_type = element::i64;

        std::vector<int32_t> sequence_pattern(elem_count);
        std::iota(sequence_pattern.begin(), sequence_pattern.end(), 0);
        auto data_to_broadcast = opset8::Constant::create(data_elem_type, {1, elem_count}, sequence_pattern);
        auto target_shape = opset8::Constant::create(target_shape_elem_type, {4}, {2, 3, 4, 1});
        auto broadcast_node =
            std::make_shared<opset8::Broadcast>(data_to_broadcast, target_shape, op::BroadcastType::BIDIRECTIONAL);

        model = std::make_shared<Model>(OutputVector{broadcast_node}, ParameterVector{});

        manager.register_pass<ov::pass::BroadcastConstRangeReplacement>();
    }
    {
        constexpr auto elem_count = 236;
        constexpr auto data_elem_type = element::i32;
        constexpr auto target_shape_elem_type = element::i64;

        auto target_shape = opset8::Constant::create(element::i64, {4}, {2, 3, 4, 1});

        const auto target_dim_neg_index = -1;
        const auto axis_node = opset8::Constant::create(element::i32, Shape{}, {0});
        const auto target_dim_index_node = opset8::Constant::create(element::i64, Shape{}, {target_dim_neg_index});
        const auto gather_dim = std::make_shared<opset8::Gather>(target_shape, target_dim_index_node, axis_node);

        // If the corresponding target dim is 1, use the original end of range
        const auto one_dim_const = opset8::Constant::create(target_shape_elem_type, {}, {1});
        const auto dim_check_one = std::make_shared<opset8::Equal>(gather_dim, one_dim_const);

        const auto start = opset8::Constant::create(data_elem_type, {}, {0});
        const auto original_end = opset8::Constant::create(data_elem_type, {}, {elem_count});

        const auto cast_gather_dim = std::make_shared<opset8::Convert>(gather_dim, data_elem_type);
        const auto select_end = std::make_shared<opset8::Select>(dim_check_one, original_end, cast_gather_dim);

        const auto default_range_step = opset8::Constant::create(data_elem_type, {}, {1});
        const auto range = std::make_shared<opset8::Range>(start, select_end, default_range_step, data_elem_type);
        const auto axes_to_unsqueeze = opset8::Constant::create(element::i64, Shape{1}, {0});
        const auto unsqueeze_range = std::make_shared<opset8::Unsqueeze>(range, axes_to_unsqueeze);

        const auto broadcast_node =
            std::make_shared<opset8::Broadcast>(unsqueeze_range, target_shape, op::BroadcastType::BIDIRECTIONAL);

        model_ref = std::make_shared<Model>(OutputVector{broadcast_node}, ParameterVector{});
    }
}

TEST_F(TransformationTestsF, BroadcastConstRangeReplacement_target_shapeof) {
    {
        constexpr auto elem_count = 236;
        constexpr auto data_elem_type = element::i32;

        std::vector<int32_t> sequence_pattern(elem_count);
        std::iota(sequence_pattern.begin(), sequence_pattern.end(), 0);

        auto data_param = std::make_shared<opset8::Parameter>(data_elem_type, Shape{2, 3, 4, elem_count});
        auto target_shape = std::make_shared<opset8::ShapeOf>(data_param);

        auto data_to_broadcast = opset8::Constant::create(data_elem_type, {1, elem_count}, sequence_pattern);
        auto broadcast_node =
            std::make_shared<opset8::Broadcast>(data_to_broadcast, target_shape, op::BroadcastType::BIDIRECTIONAL);

        model = std::make_shared<Model>(OutputVector{broadcast_node}, ParameterVector{data_param});

        manager.register_pass<ov::pass::BroadcastConstRangeReplacement>();
    }
    {
        constexpr auto elem_count = 236;
        constexpr auto data_elem_type = element::i32;
        constexpr auto target_shape_elem_type = element::i64;

        auto data_param = std::make_shared<opset8::Parameter>(data_elem_type, Shape{2, 3, 4, elem_count});
        auto target_shape = std::make_shared<opset8::ShapeOf>(data_param);

        const auto target_dim_neg_index = -1;
        const auto axis_node = opset8::Constant::create(element::i32, Shape{}, {0});
        const auto target_dim_index_node = opset8::Constant::create(element::i64, Shape{}, {target_dim_neg_index});
        const auto gather_dim = std::make_shared<opset8::Gather>(target_shape, target_dim_index_node, axis_node);

        // If the corresponding target dim is 1, use the original end of range
        const auto one_dim_const = opset8::Constant::create(target_shape_elem_type, {}, {1});
        const auto dim_check_one = std::make_shared<opset8::Equal>(gather_dim, one_dim_const);

        const auto start = opset8::Constant::create(data_elem_type, {}, {0});
        const auto original_end = opset8::Constant::create(data_elem_type, {}, {elem_count});

        const auto cast_gather_dim = std::make_shared<opset8::Convert>(gather_dim, data_elem_type);
        const auto select_end = std::make_shared<opset8::Select>(dim_check_one, original_end, cast_gather_dim);

        const auto default_range_step = opset8::Constant::create(data_elem_type, {}, {1});
        const auto range = std::make_shared<opset8::Range>(start, select_end, default_range_step, data_elem_type);
        const auto axes_to_unsqueeze = opset8::Constant::create(element::i64, Shape{1}, {0});
        const auto unsqueeze_range = std::make_shared<opset8::Unsqueeze>(range, axes_to_unsqueeze);

        const auto broadcast_node =
            std::make_shared<opset8::Broadcast>(unsqueeze_range, target_shape, op::BroadcastType::BIDIRECTIONAL);

        model_ref = std::make_shared<Model>(OutputVector{broadcast_node}, ParameterVector{data_param});
    }
}

TEST_F(TransformationTestsF, BroadcastConstRangeReplacement_target_shapeof_mixed_dims) {
    {
        constexpr auto elem_count = 236;
        constexpr auto data_elem_type = element::i32;

        std::vector<int32_t> sequence_pattern(elem_count);
        std::iota(sequence_pattern.begin(), sequence_pattern.end(), 0);

        auto data_param = std::make_shared<opset8::Parameter>(data_elem_type, Shape{2, 3, elem_count, 4});
        auto target_shape = std::make_shared<opset8::ShapeOf>(data_param);

        auto data_to_broadcast = opset8::Constant::create(data_elem_type, {1, 1, elem_count, 1}, sequence_pattern);
        auto broadcast_node =
            std::make_shared<opset8::Broadcast>(data_to_broadcast, target_shape, op::BroadcastType::BIDIRECTIONAL);

        model = std::make_shared<Model>(OutputVector{broadcast_node}, ParameterVector{data_param});

        manager.register_pass<ov::pass::BroadcastConstRangeReplacement>();
    }
    {
        constexpr auto elem_count = 236;
        constexpr auto data_elem_type = element::i32;
        constexpr auto target_shape_elem_type = element::i64;

        auto data_param = std::make_shared<opset8::Parameter>(data_elem_type, Shape{2, 3, elem_count, 4});
        auto target_shape = std::make_shared<opset8::ShapeOf>(data_param);

        const auto target_dim_neg_index = -2;
        const auto axis_node = opset8::Constant::create(element::i32, Shape{}, {0});
        const auto target_dim_index_node = opset8::Constant::create(element::i64, Shape{}, {target_dim_neg_index});
        const auto gather_dim = std::make_shared<opset8::Gather>(target_shape, target_dim_index_node, axis_node);

        const auto one_dim_const = opset8::Constant::create(target_shape_elem_type, {}, {1});
        const auto dim_check_one = std::make_shared<opset8::Equal>(gather_dim, one_dim_const);

        const auto start = opset8::Constant::create(data_elem_type, {}, {0});
        const auto original_end = opset8::Constant::create(data_elem_type, {}, {elem_count});

        const auto cast_gather_dim = std::make_shared<opset8::Convert>(gather_dim, data_elem_type);
        const auto select_end = std::make_shared<opset8::Select>(dim_check_one, original_end, cast_gather_dim);

        const auto default_range_step = opset8::Constant::create(data_elem_type, {}, {1});
        const auto range = std::make_shared<opset8::Range>(start, select_end, default_range_step, data_elem_type);

        // Axes to unsqueeze without target dim index
        const auto axes_to_unsqueeze = opset8::Constant::create(element::i64, Shape{3}, {0, 1, 3});
        const auto unsqueeze_range = std::make_shared<opset8::Unsqueeze>(range, axes_to_unsqueeze);

        const auto broadcast_node =
            std::make_shared<opset8::Broadcast>(unsqueeze_range, target_shape, op::BroadcastType::BIDIRECTIONAL);

        model_ref = std::make_shared<Model>(OutputVector{broadcast_node}, ParameterVector{data_param});
    }
}

TEST_F(TransformationTestsF, BroadcastConstRangeReplacementNeg_other_mode) {
    {
        constexpr auto elem_count = 236;
        constexpr auto data_elem_type = element::i32;
        constexpr auto target_shape_elem_type = element::i64;

        std::vector<int32_t> sequence_pattern(elem_count);
        std::iota(sequence_pattern.begin(), sequence_pattern.end(), 0);
        auto data_to_broadcast = opset8::Constant::create(data_elem_type, {1, elem_count}, sequence_pattern);
        auto target_shape = opset8::Constant::create(target_shape_elem_type, {4}, {2, 3, 4, elem_count});
        auto broadcast_node =
            std::make_shared<opset8::Broadcast>(data_to_broadcast, target_shape, op::BroadcastType::NUMPY);

        model = std::make_shared<Model>(OutputVector{broadcast_node}, ParameterVector{});

        manager.register_pass<ov::pass::BroadcastConstRangeReplacement>();
    }
}

TEST_F(TransformationTestsF, BroadcastConstRangeReplacementNeg_reversed_sequence) {
    {
        constexpr auto elem_count = 236;
        constexpr auto data_elem_type = element::i32;
        constexpr auto target_shape_elem_type = element::i64;

        std::vector<int32_t> sequence_pattern(elem_count);
        std::iota(sequence_pattern.rbegin(), sequence_pattern.rend(), 0);
        auto data_to_broadcast = opset8::Constant::create(data_elem_type, {1, elem_count}, sequence_pattern);
        auto target_shape = opset8::Constant::create(target_shape_elem_type, {4}, {2, 3, 4, elem_count});
        auto broadcast_node =
            std::make_shared<opset8::Broadcast>(data_to_broadcast, target_shape, op::BroadcastType::BIDIRECTIONAL);

        model = std::make_shared<Model>(OutputVector{broadcast_node}, ParameterVector{});

        manager.register_pass<ov::pass::BroadcastConstRangeReplacement>();
    }
}

TEST_F(TransformationTestsF, BroadcastConstRangeReplacementNeg_too_small) {
    {
        constexpr auto elem_count = 4;
        constexpr auto data_elem_type = element::i32;
        constexpr auto target_shape_elem_type = element::i64;

        std::vector<int32_t> sequence_pattern(elem_count);
        std::iota(sequence_pattern.begin(), sequence_pattern.end(), 0);
        auto data_to_broadcast = opset8::Constant::create(data_elem_type, {1, elem_count}, sequence_pattern);
        auto target_shape = opset8::Constant::create(target_shape_elem_type, {4}, {2, 3, 4, elem_count});
        auto broadcast_node =
            std::make_shared<opset8::Broadcast>(data_to_broadcast, target_shape, op::BroadcastType::NUMPY);

        model = std::make_shared<Model>(OutputVector{broadcast_node}, ParameterVector{});

        manager.register_pass<ov::pass::BroadcastConstRangeReplacement>();
    }
}

TEST_F(TransformationTestsF, BroadcastConstRangeReplacementNeg_too_big) {
    {
        constexpr auto elem_count = 1024;
        constexpr auto data_elem_type = element::i32;
        constexpr auto target_shape_elem_type = element::i64;

        std::vector<int32_t> sequence_pattern(elem_count);
        std::iota(sequence_pattern.begin(), sequence_pattern.end(), 0);
        auto data_to_broadcast = opset8::Constant::create(data_elem_type, {1, elem_count}, sequence_pattern);
        auto target_shape = opset8::Constant::create(target_shape_elem_type, {4}, {2, 3, 4, elem_count});
        auto broadcast_node =
            std::make_shared<opset8::Broadcast>(data_to_broadcast, target_shape, op::BroadcastType::NUMPY);

        model = std::make_shared<Model>(OutputVector{broadcast_node}, ParameterVector{});

        manager.register_pass<ov::pass::BroadcastConstRangeReplacement>();
    }
}

// Model reshape call test
TEST(SmartReshapeTests, BroadcastConstRangeReplacement_reshape) {
    {
        constexpr auto elem_count = 236;
        constexpr auto data_elem_type = element::i32;

        std::vector<int32_t> sequence_pattern(elem_count);
        std::iota(sequence_pattern.begin(), sequence_pattern.end(), 0);

        auto data_param = std::make_shared<opset8::Parameter>(data_elem_type, Shape{2, 3, 4, elem_count});
        auto target_shape = std::make_shared<opset8::ShapeOf>(data_param);

        auto data_to_broadcast = opset8::Constant::create(data_elem_type, {1, elem_count}, sequence_pattern);
        auto broadcast_node =
            std::make_shared<opset8::Broadcast>(data_to_broadcast, target_shape, op::BroadcastType::BIDIRECTIONAL);

        auto model = std::make_shared<Model>(OutputVector{broadcast_node}, ParameterVector{data_param});

        // BroadcastConstRangeReplacement is called as a part of SmartReshape
        EXPECT_NO_THROW(model->reshape(PartialShape{1, 189}));
    }
}

TEST_F(TransformationTestsF, BroadcastConstRangeReplacement_1D_constant) {
    {
        constexpr auto elem_count = 336;
        constexpr auto data_elem_type = element::i32;
        constexpr auto target_shape_elem_type = element::i64;

        std::vector<int32_t> sequence_pattern(elem_count);
        std::iota(sequence_pattern.begin(), sequence_pattern.end(), 0);
        auto data_to_broadcast = opset8::Constant::create(data_elem_type, {elem_count}, sequence_pattern);
        auto target_shape = opset8::Constant::create(target_shape_elem_type, {3}, {128, 8, 336});
        auto broadcast_node =
            std::make_shared<opset8::Broadcast>(data_to_broadcast, target_shape, op::BroadcastType::BIDIRECTIONAL);

        model = std::make_shared<Model>(OutputVector{broadcast_node}, ParameterVector{});

        manager.register_pass<ov::pass::BroadcastConstRangeReplacement>();
    }
    {
        constexpr auto elem_count = 336;
        constexpr auto data_elem_type = element::i32;
        constexpr auto target_shape_elem_type = element::i64;

        auto target_shape = opset8::Constant::create(element::i64, {3}, {128, 8, elem_count});

        const auto target_dim_neg_index = -1;
        const auto axis_node = opset8::Constant::create(element::i32, Shape{}, {0});
        const auto target_dim_index_node = opset8::Constant::create(element::i64, Shape{}, {target_dim_neg_index});
        const auto gather_dim = std::make_shared<opset8::Gather>(target_shape, target_dim_index_node, axis_node);

        const auto one_dim_const = opset8::Constant::create(target_shape_elem_type, {}, {1});
        const auto dim_check_one = std::make_shared<opset8::Equal>(gather_dim, one_dim_const);

        const auto start = opset8::Constant::create(data_elem_type, {}, {0});
        const auto original_end = opset8::Constant::create(data_elem_type, {}, {elem_count});

        const auto cast_gather_dim = std::make_shared<opset8::Convert>(gather_dim, data_elem_type);
        const auto select_end = std::make_shared<opset8::Select>(dim_check_one, original_end, cast_gather_dim);

        const auto default_range_step = opset8::Constant::create(data_elem_type, {}, {1});
        const auto range = std::make_shared<opset8::Range>(start, select_end, default_range_step, data_elem_type);

        const auto broadcast_node =
            std::make_shared<opset8::Broadcast>(range, target_shape, op::BroadcastType::BIDIRECTIONAL);

        model_ref = std::make_shared<Model>(OutputVector{broadcast_node}, ParameterVector{});
    }
}
