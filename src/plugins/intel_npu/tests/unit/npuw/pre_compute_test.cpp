// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "openvino/core/except.hpp"
#include "openvino/op/ops.hpp"
#include "partitioning/patterns/pre_compute.hpp"

namespace {

std::shared_ptr<ov::Model> make_longrope_v5_model(const std::vector<float>& short_factor_values,
                                                   const std::vector<float>& long_factor_values,
                                                   const std::vector<float>& multiply_values,
                                                   const std::vector<float>& power_values) {
    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 2});
    auto position_ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{2, 1});

    auto short_factor = ov::op::v0::Constant::create(ov::element::f32,
                                                      ov::Shape{short_factor_values.size()},
                                                      short_factor_values);
    auto long_factor = ov::op::v0::Constant::create(ov::element::f32,
                                                     ov::Shape{long_factor_values.size()},
                                                     long_factor_values);
    auto multiply_const = ov::op::v0::Constant::create(ov::element::f32,
                                                        ov::Shape{multiply_values.size()},
                                                        multiply_values);
    auto power_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{power_values.size()}, power_values);

    auto reduce_axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {0, 1});
    auto red_max = std::make_shared<ov::op::v1::ReduceMax>(position_ids, reduce_axes, false);
    auto one_i32 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {1});
    auto add = std::make_shared<ov::op::v1::Add>(red_max, one_i32);
    auto max_pos = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {4});
    auto greater = std::make_shared<ov::op::v1::Greater>(add, max_pos);

    auto select = std::make_shared<ov::op::v1::Select>(greater, long_factor, short_factor);
    auto multiply = std::make_shared<ov::op::v1::Multiply>(select, multiply_const);
    auto power = std::make_shared<ov::op::v1::Power>(multiply, power_const);

    auto unsqueeze_axis0 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
    auto unsq0 = std::make_shared<ov::op::v0::Unsqueeze>(power, unsqueeze_axis0);
    auto unsq1 = std::make_shared<ov::op::v0::Unsqueeze>(unsq0, unsqueeze_axis0);

    auto shape_of = std::make_shared<ov::op::v3::ShapeOf>(data);
    auto gather_idx0 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
    auto axis0 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
    auto gather = std::make_shared<ov::op::v8::Gather>(shape_of, gather_idx0, axis0);
    auto seq_len = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {4});
    auto rotary_dims = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});
    auto concat_1 = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{gather, seq_len, rotary_dims}, 0);

    auto broadcast = std::make_shared<ov::op::v3::Broadcast>(unsq1, concat_1);
    auto pos_unsq = std::make_shared<ov::op::v0::Unsqueeze>(position_ids, unsqueeze_axis0);
    auto pos_fp32 = std::make_shared<ov::op::v0::Convert>(pos_unsq, ov::element::f32);
    auto matmul = std::make_shared<ov::op::v0::MatMul>(broadcast, pos_fp32);

    auto transpose_order = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {0, 2, 1});
    auto transpose = std::make_shared<ov::op::v1::Transpose>(matmul, transpose_order);
    auto zeros = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 1, 4}, {0.0f, 0.0f, 0.0f, 0.0f});
    auto concat_2 = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{transpose, zeros}, 1);

    auto sin = std::make_shared<ov::op::v0::Sin>(concat_2);
    auto cos = std::make_shared<ov::op::v0::Cos>(concat_2);

    sin->set_friendly_name("sin_out");
    cos->set_friendly_name("cos_out");

    auto sin_res = std::make_shared<ov::op::v0::Result>(sin);
    auto cos_res = std::make_shared<ov::op::v0::Result>(cos);
    return std::make_shared<ov::Model>(ov::ResultVector{sin_res, cos_res},
                                       ov::ParameterVector{data, position_ids},
                                       "longrope_v5_test_model");
}

bool has_input_name(const std::shared_ptr<ov::Model>& model, const std::string& name) {
    const auto inputs = model->inputs();
    return std::any_of(inputs.begin(), inputs.end(), [&name](const auto& input) {
        const auto& names = input.get_names();
        return std::any_of(names.begin(), names.end(), [&name](const auto& candidate) {
            return candidate == name;
        });
    });
}

TEST(PreComputeTest, RopeCacheTransformsLongRopeV5Pattern) {
    auto model = make_longrope_v5_model({1.0f, 2.0f}, {4.0f, 5.0f}, {0.5f, 1.0f}, {2.0f});

    ov::npuw::patterns::pre_compute::RopeCache pass(/*max_prompt_len=*/16, "longrope_input");
    ASSERT_NO_THROW(pass.run_on_model(model));

    const auto& ops = model->get_ops();
    const auto sin_count = std::count_if(ops.begin(), ops.end(), [](const auto& op) {
        return ov::is_type<ov::op::v0::Sin>(op);
    });
    const auto cos_count = std::count_if(ops.begin(), ops.end(), [](const auto& op) {
        return ov::is_type<ov::op::v0::Cos>(op);
    });

    EXPECT_EQ(sin_count, 0);
    EXPECT_EQ(cos_count, 0);
    EXPECT_TRUE(has_input_name(model, "longrope_input"));
}

TEST(PreComputeTest, RopeCacheThrowsOnMismatchedFactorSizesInLongRopeV5) {
    // multiply has scalar shape {1}: graph is valid by broadcast, but calculate_freq requires exact size match.
    auto model = make_longrope_v5_model({1.0f, 2.0f}, {4.0f, 5.0f}, {1.0f}, {1.0f});
    ov::npuw::patterns::pre_compute::RopeCache pass(/*max_prompt_len=*/16, "longrope_input");

    EXPECT_THROW(pass.run_on_model(model),
                 ov::AssertFailure);
}

TEST(PreComputeTest, RopeCacheThrowsOnNonScalarPowerInLongRopeV5) {
    auto model = make_longrope_v5_model({1.0f, 2.0f}, {4.0f, 5.0f}, {1.0f, 2.0f}, {1.0f, 2.0f});
    ov::npuw::patterns::pre_compute::RopeCache pass(/*max_prompt_len=*/16, "longrope_input");

    EXPECT_THROW(pass.run_on_model(model),
                 ov::AssertFailure);
}

}  // namespace
