// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "partitioning/patterns/pre_compute.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "openvino/core/except.hpp"
#include "openvino/op/ops.hpp"
#include "orc.hpp"

namespace {

std::shared_ptr<ov::Model> make_longrope_v5_model(const std::vector<float>& short_factor_values,
                                                  const std::vector<float>& long_factor_values,
                                                  const std::vector<float>& multiply_values,
                                                  const std::vector<float>& power_values,
                                                  int32_t cond_offset = 1) {
    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 2});
    auto position_ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{2, 1});

    auto short_factor =
        ov::op::v0::Constant::create(ov::element::f32, ov::Shape{short_factor_values.size()}, short_factor_values);
    auto long_factor =
        ov::op::v0::Constant::create(ov::element::f32, ov::Shape{long_factor_values.size()}, long_factor_values);
    auto multiply_const =
        ov::op::v0::Constant::create(ov::element::f32, ov::Shape{multiply_values.size()}, multiply_values);
    auto power_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{power_values.size()}, power_values);

    auto reduce_axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {0, 1});
    auto red_max = std::make_shared<ov::op::v1::ReduceMax>(position_ids, reduce_axes, false);
    auto offset_i32 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {cond_offset});
    auto add = std::make_shared<ov::op::v1::Add>(red_max, offset_i32);
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

// Builds a minimal model matching the older LongRopePatternPhi: the Select picks
// between two ready-made inverse-frequency constants on
// max(position_ids) + 1 <= original_max_position_embeddings.
std::shared_ptr<ov::Model> make_longrope_phi_model(const std::vector<float>& inv_freq_short_values,
                                                   const std::vector<float>& inv_freq_long_values,
                                                   int32_t context_limit,
                                                   int32_t cond_offset = 1) {
    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 2});
    auto position_ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{2, 1});

    auto inv_freq_short =
        ov::op::v0::Constant::create(ov::element::f32, ov::Shape{inv_freq_short_values.size()}, inv_freq_short_values);
    auto inv_freq_long =
        ov::op::v0::Constant::create(ov::element::f32, ov::Shape{inv_freq_long_values.size()}, inv_freq_long_values);

    auto reduce_axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {0, 1});
    auto red_max = std::make_shared<ov::op::v1::ReduceMax>(position_ids, reduce_axes, false);
    auto offset_i32 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {cond_offset});
    auto add = std::make_shared<ov::op::v1::Add>(red_max, offset_i32);
    auto limit = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {context_limit});
    auto leq = std::make_shared<ov::op::v1::LessEqual>(add, limit);

    auto select = std::make_shared<ov::op::v1::Select>(leq, inv_freq_short, inv_freq_long);

    auto unsqueeze_axis0 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
    auto unsq0 = std::make_shared<ov::op::v0::Unsqueeze>(select, unsqueeze_axis0);
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

    return std::make_shared<ov::Model>(
        ov::ResultVector{std::make_shared<ov::op::v0::Result>(sin), std::make_shared<ov::op::v0::Result>(cos)},
        ov::ParameterVector{data, position_ids},
        "longrope_phi_test_model");
}

// Builds a minimal RoPE model matching RopePatternLLama2.
// When with_concat2=true (LLama2 style): Transpose → Concat_2 → Sin/Cos, duplicate_freqs=true.
// When with_concat2=false (GPT style):   Transpose → Sin/Cos directly, duplicate_freqs=false.
std::shared_ptr<ov::Model> make_rope_model(bool with_concat2) {
    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 4});
    auto position_ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1, 4});

    // inv_freq: constant [1, half_dim=2, 1]
    auto inv_freq = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 2, 1}, {0.5f, 0.1f});

    // ShapeOf → Gather(batch dim) → Concat_1 (broadcast target shape)
    auto shape_of = std::make_shared<ov::op::v3::ShapeOf>(data);
    auto gather_idx = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
    auto gather_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
    auto gather = std::make_shared<ov::op::v8::Gather>(shape_of, gather_idx, gather_axis);
    auto ndims_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});
    auto one_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto concat_1 = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{gather, ndims_const, one_const}, 0);

    // Broadcast inv_freq to [1,2,1], MatMul with position_ids → Transpose
    auto broadcast = std::make_shared<ov::op::v3::Broadcast>(inv_freq, concat_1);
    auto unsq_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(position_ids, unsq_axis);
    auto convert = std::make_shared<ov::op::v0::Convert>(unsqueeze, ov::element::f32);
    auto matmul = std::make_shared<ov::op::v0::MatMul>(broadcast, convert);  // [1,2,4]
    auto perm = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{0, 2, 1});
    auto transpose = std::make_shared<ov::op::v1::Transpose>(matmul, perm);  // [1,4,2]

    // Sin/Cos either via Concat_2 (LLama2) or directly (GPT)
    ov::Output<ov::Node> sin_cos_input = transpose->output(0);
    if (with_concat2) {
        auto zeros = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 4, 2}, std::vector<float>(8, 0.f));
        sin_cos_input = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{transpose, zeros}, -1)->output(0);
    }

    auto sin = std::make_shared<ov::op::v0::Sin>(sin_cos_input);
    auto cos = std::make_shared<ov::op::v0::Cos>(sin_cos_input);

    return std::make_shared<ov::Model>(
        ov::ResultVector{std::make_shared<ov::op::v0::Result>(sin), std::make_shared<ov::op::v0::Result>(cos)},
        ov::ParameterVector{data, position_ids},
        with_concat2 ? "llama2_rope_test_model" : "gpt_rope_test_model");
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

size_t count_ops_of_type_sin_cos(const std::shared_ptr<ov::Model>& model) {
    const auto& ops = model->get_ops();
    return static_cast<size_t>(std::count_if(ops.begin(), ops.end(), [](const auto& op) {
        return ov::is_type<ov::op::v0::Sin>(op) || ov::is_type<ov::op::v0::Cos>(op);
    }));
}

// Every LongRoPE model, old or new pattern, must end up with the same shape of result:
// no Sin/Cos and no Select left, and two named cos/sin inputs instead.
void expect_longrope_lut_inputs(const std::shared_ptr<ov::Model>& model) {
    EXPECT_EQ(count_ops_of_type_sin_cos(model), 0u);
    const auto& ops = model->get_ops();
    EXPECT_EQ(std::count_if(ops.begin(),
                            ops.end(),
                            [](const auto& op) {
                                return ov::is_type<ov::op::v1::Select>(op);
                            }),
              0)
        << "the short/long Select is decided on the host now";
    EXPECT_TRUE(has_input_name(model, ov::npuw::patterns::pre_compute::longrope_cos_input));
    EXPECT_TRUE(has_input_name(model, ov::npuw::patterns::pre_compute::longrope_sin_input));
    EXPECT_FALSE(has_input_name(model, "npuw_longrope_input"));
}

TEST(PreComputeTest, RopeCacheTransformsLongRopeV5Pattern) {
    auto model = make_longrope_v5_model({1.0f, 2.0f}, {4.0f, 5.0f}, {0.5f, 1.0f}, {2.0f});

    ov::npuw::patterns::pre_compute::RopeCache pass(/*max_prompt_len=*/16);
    ASSERT_NO_THROW(pass.run_on_model(model));

    expect_longrope_lut_inputs(model);

    const auto& tables = pass.host_tables();
    EXPECT_EQ(tables.max_len, 16u);
    EXPECT_EQ(tables.rotary_ndims, 4u);  // 2 factors, mirrored
    // short = (factor * multiply) ^ power, long likewise
    EXPECT_EQ(tables.inv_freq_short, (std::vector<float>{0.25f, 4.0f}));
    EXPECT_EQ(tables.inv_freq_long, (std::vector<float>{4.0f, 25.0f}));
}

TEST(PreComputeTest, RopeCacheTransformsLongRopePhiPattern) {
    auto model = make_longrope_phi_model({0.5f, 0.25f}, {0.1f, 0.05f}, /*context_limit=*/4096);

    ov::npuw::patterns::pre_compute::RopeCache pass(/*max_prompt_len=*/16);
    ASSERT_NO_THROW(pass.run_on_model(model));

    expect_longrope_lut_inputs(model);

    const auto& tables = pass.host_tables();
    EXPECT_EQ(tables.max_len, 16u);
    EXPECT_EQ(tables.rotary_ndims, 4u);
    // The old pattern's constants already are the inverse frequencies.
    EXPECT_EQ(tables.inv_freq_short, (std::vector<float>{0.5f, 0.25f}));
    EXPECT_EQ(tables.inv_freq_long, (std::vector<float>{0.1f, 0.05f}));
}

TEST(PreComputeTest, ExtractLongRopeContextLimitFromBothPatterns) {
    namespace pc = ov::npuw::patterns::pre_compute;

    auto v5 = make_longrope_v5_model({1.0f, 2.0f}, {4.0f, 5.0f}, {0.5f, 1.0f}, {2.0f});
    EXPECT_EQ(pc::extract_longrope_context_limit(v5), std::optional<uint64_t>{4u});

    auto phi = make_longrope_phi_model({0.5f, 0.25f}, {0.1f, 0.05f}, /*context_limit=*/4096);
    EXPECT_EQ(pc::extract_longrope_context_limit(phi), std::optional<uint64_t>{4096u});

    auto plain = make_rope_model(/*with_concat2=*/true);
    EXPECT_FALSE(pc::extract_longrope_context_limit(plain).has_value());
}

// Both regimes live in one tensor, short rows first. Views must be dense and, when the
// long half is absent, both regimes must resolve to the same rows.
TEST(PreComputeTest, LongRopeCosSinTableLayout) {
    ov::npuw::patterns::pre_compute::LongRopeCosSin tables;
    tables.max_len = 8;
    tables.rotary_ndims = 4;
    tables.inv_freq_short = {0.5f, 0.25f};
    tables.inv_freq_long = {0.1f, 0.05f};

    tables.has_long = true;
    tables.rebuild_tables();
    ASSERT_TRUE(tables.is_valid());
    EXPECT_EQ(tables.cos.get_shape(), (ov::Shape{1, 16, 4}));
    EXPECT_EQ(tables.sin.get_shape(), (ov::Shape{1, 16, 4}));

    auto short_rows = tables.cos_rows(/*lut_len=*/5, /*is_long=*/false);
    auto long_rows = tables.cos_rows(/*lut_len=*/5, /*is_long=*/true);
    EXPECT_EQ(short_rows.get_shape(), (ov::Shape{1, 5, 4}));
    EXPECT_TRUE(short_rows.is_continuous());
    EXPECT_TRUE(long_rows.is_continuous());
    EXPECT_EQ(short_rows.data<ov::float16>(), tables.cos.data<ov::float16>());
    EXPECT_EQ(long_rows.data<ov::float16>(), tables.cos.data<ov::float16>() + 8 * 4);
    // Row 1 differs between regimes because the frequencies do.
    EXPECT_NE(static_cast<float>(short_rows.data<ov::float16>()[4]),
              static_cast<float>(long_rows.data<ov::float16>()[4]));

    // Long half dropped: half the memory, and both regimes bind the short rows.
    tables.has_long = false;
    tables.rebuild_tables();
    ASSERT_TRUE(tables.is_valid());
    EXPECT_EQ(tables.cos.get_shape(), (ov::Shape{1, 8, 4}));
    EXPECT_EQ(tables.cos_rows(5, true).data<ov::float16>(), tables.cos_rows(5, false).data<ov::float16>());
}

// The host re-evaluates the regime as max(position_ids) >= context_limit, which only
// matches the graph when the compared sum adds exactly 1. Anything else must be
// refused instead of silently binding the wrong coefficients.
TEST(PreComputeTest, RopeCacheRejectsUnexpectedConditionOffset) {
    namespace pc = ov::npuw::patterns::pre_compute;

    auto v5 = make_longrope_v5_model({1.0f, 2.0f}, {4.0f, 5.0f}, {0.5f, 1.0f}, {2.0f}, /*cond_offset=*/2);
    EXPECT_THROW(pc::extract_longrope_context_limit(v5), ov::AssertFailure);
    EXPECT_THROW(pc::RopeCache(/*max_prompt_len=*/16).run_on_model(v5), ov::AssertFailure);

    auto phi = make_longrope_phi_model({0.5f, 0.25f}, {0.1f, 0.05f}, /*context_limit=*/4096, /*cond_offset=*/0);
    EXPECT_THROW(pc::extract_longrope_context_limit(phi), ov::AssertFailure);
    EXPECT_THROW(pc::RopeCache(/*max_prompt_len=*/16).run_on_model(phi), ov::AssertFailure);
}

// The npuw_lr_cos/npuw_lr_sin inputs of an imported blob are bound to tables that are
// regenerated from the serialized metadata, so a round trip must reproduce them exactly.
TEST(PreComputeTest, LongRopeCosSinSerializationRoundTrip) {
    using ov::npuw::orc::Stream;

    for (bool has_long : {true, false}) {
        ov::npuw::patterns::pre_compute::LongRopeCosSin src;
        src.max_len = 12;
        src.rotary_ndims = 4;
        src.has_long = has_long;
        src.inv_freq_short = {0.5f, 0.25f};
        src.inv_freq_long = {0.1f, 0.05f};
        src.rebuild_tables();
        ASSERT_TRUE(src.is_valid());

        std::stringstream ss;
        auto writer = Stream::writer(ss);
        writer& src;

        ov::npuw::patterns::pre_compute::LongRopeCosSin dst;
        auto reader = Stream::reader(ss);
        reader& dst;

        EXPECT_EQ(dst.max_len, src.max_len);
        EXPECT_EQ(dst.rotary_ndims, src.rotary_ndims);
        EXPECT_EQ(dst.has_long, src.has_long);
        EXPECT_EQ(dst.inv_freq_short, src.inv_freq_short);
        EXPECT_EQ(dst.inv_freq_long, src.inv_freq_long);

        // serialize() rebuilds the tensors on read - they must come back byte-identical.
        ASSERT_TRUE(dst.is_valid()) << "imported tables must be usable without re-running RopeCache";
        ASSERT_EQ(dst.cos.get_shape(), src.cos.get_shape());
        ASSERT_EQ(dst.sin.get_shape(), src.sin.get_shape());
        EXPECT_EQ(std::memcmp(dst.cos.data(), src.cos.data(), src.cos.get_byte_size()), 0);
        EXPECT_EQ(std::memcmp(dst.sin.data(), src.sin.data(), src.sin.get_byte_size()), 0);

        EXPECT_EQ(dst.cos_rows(5, true).get_shape(), (ov::Shape{1, 5, 4}));
        EXPECT_EQ(dst.cos_rows(5, true).data<ov::float16>() == dst.cos_rows(5, false).data<ov::float16>(), !has_long);
    }
}

// A model with no LongRoPE leaves the tables empty; that must survive a round trip too,
// and must not fabricate tensors on import.
TEST(PreComputeTest, LongRopeCosSinSerializationRoundTripEmpty) {
    using ov::npuw::orc::Stream;

    ov::npuw::patterns::pre_compute::LongRopeCosSin src;
    std::stringstream ss;
    auto writer = Stream::writer(ss);
    writer& src;

    ov::npuw::patterns::pre_compute::LongRopeCosSin dst;
    dst.max_len = 7;  // must be overwritten by the read
    auto reader = Stream::reader(ss);
    reader& dst;

    EXPECT_EQ(dst.max_len, 0u);
    EXPECT_EQ(dst.rotary_ndims, 0u);
    EXPECT_FALSE(dst.has_long);
    EXPECT_FALSE(dst.is_valid());
}

TEST(PreComputeTest, RopeCacheThrowsOnMismatchedFactorSizesInLongRopeV5) {
    // multiply has scalar shape {1}: graph is valid by broadcast, but calculate_freq requires exact size match.
    auto model = make_longrope_v5_model({1.0f, 2.0f}, {4.0f, 5.0f}, {1.0f}, {1.0f});
    ov::npuw::patterns::pre_compute::RopeCache pass(/*max_prompt_len=*/16);

    EXPECT_THROW(pass.run_on_model(model), ov::AssertFailure);
}

TEST(PreComputeTest, RopeCacheThrowsOnNonScalarPowerInLongRopeV5) {
    auto model = make_longrope_v5_model({1.0f, 2.0f}, {4.0f, 5.0f}, {1.0f, 2.0f}, {1.0f, 2.0f});
    ov::npuw::patterns::pre_compute::RopeCache pass(/*max_prompt_len=*/16);

    EXPECT_THROW(pass.run_on_model(model), ov::AssertFailure);
}

// Verifies that the merged RopePatternLLama2 correctly detects and removes the
// LLama2-style sin/cos subgraph (with Concat_2 present, duplicate_freqs=true).
TEST(PreComputeTest, RopeCacheTransformsLLama2Pattern) {
    auto model = make_rope_model(/*with_concat2=*/true);

    ov::npuw::patterns::pre_compute::RopeCache pass(/*max_prompt_len=*/16);
    ASSERT_NO_THROW(pass.run_on_model(model));

    const auto& ops = model->get_ops();
    const auto sin_count = std::count_if(ops.begin(), ops.end(), [](const auto& op) {
        return ov::is_type<ov::op::v0::Sin>(op);
    });
    const auto cos_count = std::count_if(ops.begin(), ops.end(), [](const auto& op) {
        return ov::is_type<ov::op::v0::Cos>(op);
    });

    EXPECT_EQ(sin_count, 0) << "Sin should be replaced by Gather from the duplicated LUT";
    EXPECT_EQ(cos_count, 0) << "Cos should be replaced by Gather from the duplicated LUT";
}

// Verifies that the merged RopePatternLLama2 correctly detects and removes the
// GPT-style sin/cos subgraph (Concat_2 absent, duplicate_freqs=false).
TEST(PreComputeTest, RopeCacheTransformsGPTPattern) {
    auto model = make_rope_model(/*with_concat2=*/false);

    ov::npuw::patterns::pre_compute::RopeCache pass(/*max_prompt_len=*/16);
    ASSERT_NO_THROW(pass.run_on_model(model));

    const auto& ops = model->get_ops();
    const auto sin_count = std::count_if(ops.begin(), ops.end(), [](const auto& op) {
        return ov::is_type<ov::op::v0::Sin>(op);
    });
    const auto cos_count = std::count_if(ops.begin(), ops.end(), [](const auto& op) {
        return ov::is_type<ov::op::v0::Cos>(op);
    });

    EXPECT_EQ(sin_count, 0) << "Sin should be replaced by Gather from the non-duplicated LUT";
    EXPECT_EQ(cos_count, 0) << "Cos should be replaced by Gather from the non-duplicated LUT";
}

}  // namespace
