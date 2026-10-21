// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "partitioning/patterns/pre_compute.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "openvino/core/except.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"
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

    // Sin/Cos either via Concat_2 (LLama2's torch.cat([freqs, freqs], -1), i.e. transpose
    // concatenated with itself) or directly (GPT)
    ov::Output<ov::Node> sin_cos_input = transpose->output(0);
    if (with_concat2) {
        sin_cos_input = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{transpose, transpose}, -1)->output(0);
    }

    auto sin = std::make_shared<ov::op::v0::Sin>(sin_cos_input);
    auto cos = std::make_shared<ov::op::v0::Cos>(sin_cos_input);

    return std::make_shared<ov::Model>(
        ov::ResultVector{std::make_shared<ov::op::v0::Result>(sin), std::make_shared<ov::op::v0::Result>(cos)},
        ov::ParameterVector{data, position_ids},
        with_concat2 ? "llama2_rope_test_model" : "gpt_rope_test_model");
}

// Same as make_rope_model(/*with_concat2=*/true), but the Concat's 2nd input is zeros instead
// of a duplicate of the transpose output - e.g. partial-rotary models that pad rather than
// duplicate. The matcher must not mistake this for freq duplication.
std::shared_ptr<ov::Model> make_rope_model_non_duplicating_concat() {
    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 4});
    auto position_ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1, 4});

    auto inv_freq = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 2, 1}, {0.5f, 0.1f});

    auto shape_of = std::make_shared<ov::op::v3::ShapeOf>(data);
    auto gather_idx = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
    auto gather_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
    auto gather = std::make_shared<ov::op::v8::Gather>(shape_of, gather_idx, gather_axis);
    auto ndims_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});
    auto one_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto concat_1 = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{gather, ndims_const, one_const}, 0);

    auto broadcast = std::make_shared<ov::op::v3::Broadcast>(inv_freq, concat_1);
    auto unsq_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(position_ids, unsq_axis);
    auto convert = std::make_shared<ov::op::v0::Convert>(unsqueeze, ov::element::f32);
    auto matmul = std::make_shared<ov::op::v0::MatMul>(broadcast, convert);
    auto perm = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{0, 2, 1});
    auto transpose = std::make_shared<ov::op::v1::Transpose>(matmul, perm);

    auto zeros = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 4, 2}, std::vector<float>(8, 0.f));
    auto concat_2 = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{transpose, zeros}, -1);

    auto sin = std::make_shared<ov::op::v0::Sin>(concat_2);
    auto cos = std::make_shared<ov::op::v0::Cos>(concat_2);

    return std::make_shared<ov::Model>(
        ov::ResultVector{std::make_shared<ov::op::v0::Result>(sin), std::make_shared<ov::op::v0::Result>(cos)},
        ov::ParameterVector{data, position_ids},
        "non_duplicating_concat_rope_test_model");
}

// Builds a RoPE model matching make_rope_model's two flavours, but where inv_freq is
// already exported pre-broadcast to its final [1, half_dim, 1] shape, with no live
// Broadcast node (and no ShapeOf->Gather->Concat shape-compute chain feeding it) - as
// seen e.g. on Qwen3, where the exporter folds the broadcast away.
std::shared_ptr<ov::Model> make_rope_model_no_broadcast(bool with_concat2) {
    auto position_ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1, 4});

    // inv_freq already shaped as if Broadcast had run: [1, half_dim=2, 1]
    auto inv_freq = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 2, 1}, {0.5f, 0.1f});

    auto unsq_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(position_ids, unsq_axis);
    auto convert = std::make_shared<ov::op::v0::Convert>(unsqueeze, ov::element::f32);
    auto matmul = std::make_shared<ov::op::v0::MatMul>(inv_freq, convert);  // [1,2,4]
    auto perm = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{0, 2, 1});
    auto transpose = std::make_shared<ov::op::v1::Transpose>(matmul, perm);  // [1,4,2]

    ov::Output<ov::Node> sin_cos_input = transpose->output(0);
    if (with_concat2) {
        sin_cos_input = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{transpose, transpose}, -1)->output(0);
    }

    auto sin = std::make_shared<ov::op::v0::Sin>(sin_cos_input);
    auto cos = std::make_shared<ov::op::v0::Cos>(sin_cos_input);

    return std::make_shared<ov::Model>(
        ov::ResultVector{std::make_shared<ov::op::v0::Result>(sin), std::make_shared<ov::op::v0::Result>(cos)},
        ov::ParameterVector{position_ids},
        with_concat2 ? "llama2_rope_no_broadcast_test_model" : "gpt_rope_no_broadcast_test_model");
}

// Same as make_rope_model_no_broadcast(/*with_concat2=*/true), but inv_freq has a real (non-
// singleton) leading batch dim - a shape the "no live Broadcast" fallback never validates.
// Without a validity check, makeCosSinCache would flatten this into the frequency axis and
// silently corrupt the LUT's shape, so the matcher must leave Sin/Cos untouched instead.
std::shared_ptr<ov::Model> make_rope_model_no_broadcast_extra_batch_dim() {
    auto position_ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1, 4});

    // inv_freq shaped [2, 2, 1] - batch=2, not the expected singleton batch dim.
    auto inv_freq = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{2, 2, 1}, {0.5f, 0.1f, 0.5f, 0.1f});

    auto unsq_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(position_ids, unsq_axis);
    auto convert = std::make_shared<ov::op::v0::Convert>(unsqueeze, ov::element::f32);
    auto matmul = std::make_shared<ov::op::v0::MatMul>(inv_freq, convert);  // [2,2,4]
    auto perm = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{0, 2, 1});
    auto transpose = std::make_shared<ov::op::v1::Transpose>(matmul, perm);  // [2,4,2]

    auto concat_2 = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{transpose, transpose}, -1);

    auto sin = std::make_shared<ov::op::v0::Sin>(concat_2);
    auto cos = std::make_shared<ov::op::v0::Cos>(concat_2);

    return std::make_shared<ov::Model>(
        ov::ResultVector{std::make_shared<ov::op::v0::Result>(sin), std::make_shared<ov::op::v0::Result>(cos)},
        ov::ParameterVector{position_ids},
        "llama2_rope_no_broadcast_extra_batch_dim_test_model");
}

// Builds a Gemma 4-style model with hybrid attention, it has multiple independent RoPE blocks.
// `result_order`, as (block index, is_cos) pairs, lets a test pick a non-default Results
// ordering; empty means the natural per-block (sin_i, cos_i) grouping.
std::shared_ptr<ov::Model> make_multi_rope_model(const std::vector<std::vector<float>>& inv_freqs,
                                                 const std::vector<std::pair<size_t, bool>>& result_order = {}) {
    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 4});
    auto position_ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1, 4});

    // Shared shape-compute chain (ShapeOf->Gather->Concat_1).
    auto shape_of = std::make_shared<ov::op::v3::ShapeOf>(data);
    auto gather_idx = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
    auto gather_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
    auto gather = std::make_shared<ov::op::v8::Gather>(shape_of, gather_idx, gather_axis);
    auto ndims_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});
    auto one_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto concat_1 = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{gather, ndims_const, one_const}, 0);

    // Shared position_ids branch (Unsqueeze->Convert).
    auto unsq_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(position_ids, unsq_axis);
    auto convert = std::make_shared<ov::op::v0::Convert>(unsqueeze, ov::element::f32);

    std::vector<ov::Output<ov::Node>> sin_outs(inv_freqs.size());
    std::vector<ov::Output<ov::Node>> cos_outs(inv_freqs.size());
    for (size_t i = 0; i < inv_freqs.size(); ++i) {
        const auto& freqs = inv_freqs[i];
        auto inv_freq = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, freqs.size(), 1}, freqs);

        auto broadcast = std::make_shared<ov::op::v3::Broadcast>(inv_freq, concat_1);
        auto matmul = std::make_shared<ov::op::v0::MatMul>(broadcast, convert);  // [1, freqs.size(), 4]
        auto perm = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{0, 2, 1});
        auto transpose = std::make_shared<ov::op::v1::Transpose>(matmul, perm);  // [1, 4, freqs.size()]

        // torch.cat([freqs, freqs], -1): the transpose output concatenated with itself.
        auto concat_2 = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{transpose, transpose}, -1);

        auto sin = std::make_shared<ov::op::v0::Sin>(concat_2);
        auto cos = std::make_shared<ov::op::v0::Cos>(concat_2);
        sin->set_friendly_name("sin_" + std::to_string(i));
        cos->set_friendly_name("cos_" + std::to_string(i));

        sin_outs[i] = sin->output(0);
        cos_outs[i] = cos->output(0);
    }

    ov::ResultVector results;
    if (result_order.empty()) {
        for (size_t i = 0; i < inv_freqs.size(); ++i) {
            results.push_back(std::make_shared<ov::op::v0::Result>(sin_outs[i]));
            results.push_back(std::make_shared<ov::op::v0::Result>(cos_outs[i]));
        }
    } else {
        for (const auto& entry : result_order) {
            results.push_back(
                std::make_shared<ov::op::v0::Result>(entry.second ? cos_outs[entry.first] : sin_outs[entry.first]));
        }
    }

    return std::make_shared<ov::Model>(results, ov::ParameterVector{data, position_ids}, "multi_rope_test_model");
}

// Builds an N-section M-RoPE model (Qwen2-VL-style multi-section rotary): N stacked rows of
// position_ids share one inv_freq, producing one batched [N,batch,seq,width] frequency tensor;
// row 0 seeds a [batch,seq,width] accumulator and rows 1..N-1 are each sliced to their own
// disjoint channel sub-range and merged in via ScatterNDUpdate, before the final
// self-duplicating Concat feeding Sin/Cos. `section_width` is the per-section channel width
// (equal for every section, including row 0, whose implicit/complement range is
// [0, section_width)). `num_sections` is intentionally a runtime parameter, not a fixed
// constant, so tests can verify the pattern isn't hardcoded to any specific section count.
std::shared_ptr<ov::Model> make_mrope_model(int64_t num_sections, int64_t section_width) {
    const int64_t batch = 1;
    const int64_t seq = 4;
    const int64_t full_width = num_sections * section_width;

    auto position_ids = std::make_shared<ov::op::v0::Parameter>(
        ov::element::i64,
        ov::Shape{static_cast<size_t>(num_sections), static_cast<size_t>(batch), static_cast<size_t>(seq)});

    std::vector<float> inv_freq_values(static_cast<size_t>(full_width));
    for (size_t k = 0; k < inv_freq_values.size(); ++k) {
        inv_freq_values[k] = 0.1f + 0.05f * static_cast<float>(k);
    }
    // Pre-broadcast shaped (singleton N and batch dims) - MatMul's own numpy-style batch
    // broadcasting fans it out against position_ids' real N/batch dims below, same convention
    // as RopePatternLLama2's "no broadcast chain" test models (no live Broadcast op).
    auto inv_freq = ov::op::v0::Constant::create(ov::element::f32,
                                                 ov::Shape{1, 1, static_cast<size_t>(full_width), 1},
                                                 inv_freq_values);

    auto unsq_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});
    auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(position_ids, unsq_axis);  // [N,batch,1,seq]
    auto convert = std::make_shared<ov::op::v0::Convert>(unsqueeze, ov::element::f32);
    auto matmul = std::make_shared<ov::op::v0::MatMul>(inv_freq, convert);  // [N,batch,width,seq]
    auto perm = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 1, 3, 2});
    auto transpose = std::make_shared<ov::op::v1::Transpose>(matmul, perm);  // [N,batch,seq,width]

    auto make_row_index = [](int64_t row) {
        return ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {row});
    };
    auto axis0 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});

    // Row 0: the implicit/terminal section, read directly off `transpose` with no slicing - its
    // effective channel range ([0, section_width)) is never explicitly scattered over.
    ov::Output<ov::Node> accumulator =
        std::make_shared<ov::op::v8::Gather>(transpose, make_row_index(0), axis0)->output(0);  // [batch,seq,width]

    for (int64_t i = 1; i < num_sections; ++i) {
        auto select = std::make_shared<ov::op::v8::Gather>(transpose, make_row_index(i), axis0);  // [batch,seq,width]
        auto begin = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {i * section_width});
        auto end = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {(i + 1) * section_width});
        auto step = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
        auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
        auto slice = std::make_shared<ov::op::v8::Slice>(select, begin, end, step, axis);  // [batch,seq,width_i]

        // ScatterNDUpdate indices: one [b,s,c] coordinate triple per element of `slice`, placing
        // it at the same channel offset in `accumulator` it was sliced from in `select`.
        std::vector<int64_t> indices_values;
        indices_values.reserve(static_cast<size_t>(batch * seq * section_width * 3));
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t s = 0; s < seq; ++s) {
                for (int64_t c = 0; c < section_width; ++c) {
                    indices_values.insert(indices_values.end(), {b, s, i * section_width + c});
                }
            }
        }
        auto indices = ov::op::v0::Constant::create(
            ov::element::i64,
            ov::Shape{static_cast<size_t>(batch), static_cast<size_t>(seq), static_cast<size_t>(section_width), 3},
            indices_values);
        accumulator = std::make_shared<ov::op::v3::ScatterNDUpdate>(accumulator, indices, slice)->output(0);
    }

    auto concat_2 = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{accumulator, accumulator}, -1);
    auto sin = std::make_shared<ov::op::v0::Sin>(concat_2);
    auto cos = std::make_shared<ov::op::v0::Cos>(concat_2);

    return std::make_shared<ov::Model>(
        ov::ResultVector{std::make_shared<ov::op::v0::Result>(sin), std::make_shared<ov::op::v0::Result>(cos)},
        ov::ParameterVector{position_ids},
        "mrope_test_model_" + std::to_string(num_sections) + "_sections");
}

// Builds an N-section M-RoPE model using an INTERLEAVED (strided) channel assignment instead of
// contiguous blocks - section i (i=1..N-1) claims channels {i, i+N, i+2N, ...} < full_width, so
// row 0's implicit/leftover channels are {0, N, 2N, ...} too. Mirrors the "interleaved mrope"
// layout real-world models (e.g. Qwen3-VL-style) use instead of a simple [0,w)/[w,2w)/... split -
// the real bug this guards against: a sorted-Slice-then-Concat reassembly of contiguous ranges
// cannot represent a channel_step > 1 section, only a ScatterUpdate-based merge can.
std::shared_ptr<ov::Model> make_mrope_model_interleaved(int64_t num_sections, int64_t full_width) {
    const int64_t batch = 1;
    const int64_t seq = 4;

    auto position_ids = std::make_shared<ov::op::v0::Parameter>(
        ov::element::i64,
        ov::Shape{static_cast<size_t>(num_sections), static_cast<size_t>(batch), static_cast<size_t>(seq)});

    std::vector<float> inv_freq_values(static_cast<size_t>(full_width));
    for (size_t k = 0; k < inv_freq_values.size(); ++k) {
        inv_freq_values[k] = 0.1f + 0.05f * static_cast<float>(k);
    }
    auto inv_freq = ov::op::v0::Constant::create(ov::element::f32,
                                                 ov::Shape{1, 1, static_cast<size_t>(full_width), 1},
                                                 inv_freq_values);

    auto unsq_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});
    auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(position_ids, unsq_axis);  // [N,batch,1,seq]
    auto convert = std::make_shared<ov::op::v0::Convert>(unsqueeze, ov::element::f32);
    auto matmul = std::make_shared<ov::op::v0::MatMul>(inv_freq, convert);  // [N,batch,width,seq]
    auto perm = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 1, 3, 2});
    auto transpose = std::make_shared<ov::op::v1::Transpose>(matmul, perm);  // [N,batch,seq,width]

    auto make_row_index = [](int64_t row) {
        return ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {row});
    };
    auto axis0 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});

    ov::Output<ov::Node> accumulator =
        std::make_shared<ov::op::v8::Gather>(transpose, make_row_index(0), axis0)->output(0);  // [batch,seq,width]

    for (int64_t i = 1; i < num_sections; ++i) {
        auto select = std::make_shared<ov::op::v8::Gather>(transpose, make_row_index(i), axis0);  // [batch,seq,width]

        // Interleaved: channels {i, i+num_sections, i+2*num_sections, ...} < full_width.
        int64_t count = 0;
        for (int64_t c = i; c < full_width; c += num_sections) {
            ++count;
        }
        auto begin = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {i});
        auto end = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {full_width});
        auto step = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {num_sections});
        auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
        auto slice = std::make_shared<ov::op::v8::Slice>(select, begin, end, step, axis);  // [batch,seq,count]

        std::vector<int64_t> indices_values;
        indices_values.reserve(static_cast<size_t>(batch * seq * count * 3));
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t s = 0; s < seq; ++s) {
                for (int64_t c = i; c < full_width; c += num_sections) {
                    indices_values.insert(indices_values.end(), {b, s, c});
                }
            }
        }
        auto indices = ov::op::v0::Constant::create(
            ov::element::i64,
            ov::Shape{static_cast<size_t>(batch), static_cast<size_t>(seq), static_cast<size_t>(count), 3},
            indices_values);
        accumulator = std::make_shared<ov::op::v3::ScatterNDUpdate>(accumulator, indices, slice)->output(0);
    }

    auto concat_2 = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{accumulator, accumulator}, -1);
    auto sin = std::make_shared<ov::op::v0::Sin>(concat_2);
    auto cos = std::make_shared<ov::op::v0::Cos>(concat_2);

    return std::make_shared<ov::Model>(
        ov::ResultVector{std::make_shared<ov::op::v0::Result>(sin), std::make_shared<ov::op::v0::Result>(cos)},
        ov::ParameterVector{position_ids},
        "mrope_test_model_interleaved_" + std::to_string(num_sections) + "_sections");
}

// A model with one full RoPE block (Sin+Cos) plus a second, independent Cos with no Sin partner.
std::shared_ptr<ov::Model> make_rope_model_with_lone_cos(const std::vector<float>& paired_freqs,
                                                         const std::vector<float>& lone_cos_freqs) {
    auto model = make_multi_rope_model({paired_freqs, lone_cos_freqs});

    // Drop the second block's Sin result, turning it into a Cos-only chain.
    auto results = model->get_results();
    model->remove_result(results[2]);  // sin_1
    return model;
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

// Both modes live in one tensor, short rows first. Views must be dense and, when the
// long half is absent, both modes must resolve to the same rows.
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
    // Row 1 differs between modes because the frequencies do.
    EXPECT_NE(static_cast<float>(short_rows.data<ov::float16>()[4]),
              static_cast<float>(long_rows.data<ov::float16>()[4]));

    // Long half dropped: half the memory, and both modes bind the short rows.
    tables.has_long = false;
    tables.rebuild_tables();
    ASSERT_TRUE(tables.is_valid());
    EXPECT_EQ(tables.cos.get_shape(), (ov::Shape{1, 8, 4}));
    EXPECT_EQ(tables.cos_rows(5, true).data<ov::float16>(), tables.cos_rows(5, false).data<ov::float16>());
}

// The host re-evaluates the mode as max(position_ids) >= context_limit, which only
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
        writer & src;

        ov::npuw::patterns::pre_compute::LongRopeCosSin dst;
        auto reader = Stream::reader(ss);
        reader & dst;

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
    writer & src;

    ov::npuw::patterns::pre_compute::LongRopeCosSin dst;
    dst.max_len = 7;  // must be overwritten by the read
    auto reader = Stream::reader(ss);
    reader & dst;

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

// Qwen3-style export: inv_freq is already a pre-broadcast Constant, no live Broadcast
// node and no ShapeOf->Gather->Concat chain feeding it. The matcher must still fire via
// the optional Broadcast's fallback to matching inv_freq's output directly.
TEST(PreComputeTest, RopeCacheTransformsLLama2PatternWithoutBroadcastChain) {
    auto model = make_rope_model_no_broadcast(/*with_concat2=*/true);

    ov::npuw::patterns::pre_compute::RopeCache pass(/*max_prompt_len=*/16);
    ASSERT_NO_THROW(pass.run_on_model(model));

    EXPECT_EQ(count_ops_of_type_sin_cos(model), 0u) << "Sin/Cos should be replaced by Gather from the LUT";
}

TEST(PreComputeTest, RopeCacheTransformsGPTPatternWithoutBroadcastChain) {
    auto model = make_rope_model_no_broadcast(/*with_concat2=*/false);

    ov::npuw::patterns::pre_compute::RopeCache pass(/*max_prompt_len=*/16);
    ASSERT_NO_THROW(pass.run_on_model(model));

    EXPECT_EQ(count_ops_of_type_sin_cos(model), 0u) << "Sin/Cos should be replaced by Gather from the LUT";
}

// When the Broadcast is bypassed, inv_freq must still have only the supported singleton
// batch dim - anything else (e.g. a real batch dim) would be silently flattened into the
// frequency axis by makeCosSinCache, corrupting the LUT's shape. The matcher must leave
// Sin/Cos untouched rather than misfire on such a constant.
TEST(PreComputeTest, RopeCacheLeavesNonSingletonBatchInvFreqUntouched) {
    auto model = make_rope_model_no_broadcast_extra_batch_dim();

    ov::npuw::patterns::pre_compute::RopeCache pass(/*max_prompt_len=*/16);
    ASSERT_NO_THROW(pass.run_on_model(model));

    EXPECT_EQ(count_ops_of_type_sin_cos(model), 2u) << "Sin/Cos must stay as-is, not be wrongly replaced by the LUT";
}

// A Concat following Transpose whose 2nd input isn't the same tensor (e.g. zero-padding for
// partial-rotary models) must not be mistaken for freq duplication - the LUT's second half
// would otherwise be wrong. The matcher must leave Sin/Cos untouched rather than misfire.
TEST(PreComputeTest, RopeCacheLeavesNonDuplicatingConcatUntouched) {
    auto model = make_rope_model_non_duplicating_concat();

    ov::npuw::patterns::pre_compute::RopeCache pass(/*max_prompt_len=*/16);
    ASSERT_NO_THROW(pass.run_on_model(model));

    EXPECT_EQ(count_ops_of_type_sin_cos(model), 2u) << "Sin/Cos must stay as-is, not be wrongly replaced by the LUT";
}

// A model can contain more than one independent RoPE block (e.g. Gemma 4's hybrid attention)
// and every block must be cached, not just the first.
TEST(PreComputeTest, RopeCacheTransformsTwoIndependentRopeBlocks) {
    auto model = make_multi_rope_model({{0.5f, 0.1f}, {0.2f, 0.05f}});

    ov::npuw::patterns::pre_compute::RopeCache pass(/*max_prompt_len=*/16);
    ASSERT_NO_THROW(pass.run_on_model(model));

    EXPECT_EQ(count_ops_of_type_sin_cos(model), 0u) << "both RoPE blocks must be cached, not just the first";

    const auto& ops = model->get_ops();
    const auto gather_count = std::count_if(ops.begin(), ops.end(), [](const auto& op) {
        return ov::is_type<ov::op::v8::Gather>(op);
    });
    EXPECT_EQ(gather_count, 4u) << "one Gather for cos and one for sin, per block";

    // Each block must get its own cache, not accidentally share one.
    std::vector<const ov::op::v0::Constant*> caches;
    for (const auto& op : ops) {
        if (auto gather_op = ov::as_type_ptr<ov::op::v8::Gather>(op)) {
            if (auto cache = ov::as_type_ptr<ov::op::v0::Constant>(gather_op->input_value(0).get_node_shared_ptr())) {
                caches.push_back(cache.get());
            }
        }
    }
    ASSERT_EQ(caches.size(), 4u);
    for (size_t i = 0; i < caches.size(); ++i) {
        for (size_t j = i + 1; j < caches.size(); ++j) {
            EXPECT_NE(std::memcmp(caches[i]->get_data_ptr(), caches[j]->get_data_ptr(), caches[i]->get_byte_size()), 0)
                << "cache constants " << i << " and " << j << " are unexpectedly identical";
        }
    }
}

// Reads the f16 cache row a Result's Squeeze->Convert->Gather chain was bound to.
std::vector<float> read_cache_row(const ov::Output<ov::Node>& result_input, size_t row) {
    auto convert = result_input.get_node()->input_value(0).get_node();
    auto gather = convert->input_value(0).get_node();
    auto cache = ov::as_type_ptr<ov::op::v0::Constant>(gather->input_value(0).get_node_shared_ptr());
    OPENVINO_ASSERT(cache, "Expected a cache Constant feeding the Gather");

    const auto* data = cache->get_data_ptr<ov::float16>();
    const auto rotary_ndims = cache->get_shape().back();
    std::vector<float> values(rotary_ndims);
    for (size_t k = 0; k < rotary_ndims; ++k) {
        values[k] = static_cast<float>(data[row * rotary_ndims + k]);
    }
    return values;
}

// Match pairing must key off the shared input node, not vector index - MultiMatcher doesn't
// guarantee a block's Sin and Cos land at the same index. Results are ordered here so naive
// index-based pairing would cross-wire the two blocks' caches.
TEST(PreComputeTest, RopeCacheTransformsTwoIndependentRopeBlocksWithScrambledMatchOrder) {
    auto model = make_multi_rope_model({{0.5f, 0.1f}, {0.2f, 0.05f}},
                                       /*result_order=*/{{0, false}, {1, false}, {1, true}, {0, true}});

    ov::npuw::patterns::pre_compute::RopeCache pass(/*max_prompt_len=*/16);
    ASSERT_NO_THROW(pass.run_on_model(model));

    const auto& results = model->get_results();
    auto sin0 = read_cache_row(results[0]->input_value(0), /*row=*/1);
    auto sin1 = read_cache_row(results[1]->input_value(0), /*row=*/1);
    auto cos1 = read_cache_row(results[2]->input_value(0), /*row=*/1);
    auto cos0 = read_cache_row(results[3]->input_value(0), /*row=*/1);

    for (size_t k = 0; k < sin0.size(); ++k) {
        EXPECT_NEAR(cos0[k] * cos0[k] + sin0[k] * sin0[k], 1.0f, 0.01f) << "block 0 cos/sin mismatch at " << k;
        EXPECT_NEAR(cos1[k] * cos1[k] + sin1[k] * sin1[k], 1.0f, 0.01f) << "block 1 cos/sin mismatch at " << k;
    }
}

// A pattern match without its sin/cos counterpart (e.g. an unrelated Cos-only computation that
// coincidentally matches the same antecedent chain) must be left in place, not rejected.
TEST(PreComputeTest, RopeCacheLeavesUnpairedCosUntouched) {
    auto model = make_rope_model_with_lone_cos({0.5f, 0.1f}, {0.2f, 0.05f});

    ov::npuw::patterns::pre_compute::RopeCache pass(/*max_prompt_len=*/16);
    ASSERT_NO_THROW(pass.run_on_model(model));

    const auto& ops = model->get_ops();
    EXPECT_EQ(std::count_if(ops.begin(),
                            ops.end(),
                            [](const auto& op) {
                                return ov::is_type<ov::op::v0::Sin>(op);
                            }),
              0)
        << "the paired Sin must be cached away";
    EXPECT_EQ(std::count_if(ops.begin(),
                            ops.end(),
                            [](const auto& op) {
                                return ov::is_type<ov::op::v0::Cos>(op);
                            }),
              1)
        << "only the lone, unpaired Cos should remain";
}

// Counts ops of type T still reachable from the model's current Results.
template <typename T>
size_t count_ops_of_type(const std::shared_ptr<ov::Model>& model) {
    const auto& ops = model->get_ops();
    return static_cast<size_t>(std::count_if(ops.begin(), ops.end(), [](const auto& op) {
        return ov::is_type<T>(op);
    }));
}

// RopePatternMRope must detect and cache M-RoPE's multi-section merge idiom regardless of how
// many sections it stacks - not just a fixed/hardcoded count. Parameterized so the same checks
// run for several different section counts (see the TEST() instantiations below).
void expect_mrope_cached(int64_t num_sections, int64_t section_width) {
    auto model = make_mrope_model(num_sections, section_width);

    ov::npuw::patterns::pre_compute::RopeCache pass(/*max_prompt_len=*/16);
    ASSERT_NO_THROW(pass.run_on_model(model));

    EXPECT_EQ(count_ops_of_type<ov::op::v0::Sin>(model), 0u) << "Sin should be replaced by the merged LUT cache";
    EXPECT_EQ(count_ops_of_type<ov::op::v0::Cos>(model), 0u) << "Cos should be replaced by the merged LUT cache";
    EXPECT_EQ(count_ops_of_type<ov::op::v3::ScatterNDUpdate>(model), 0u)
        << "the now-dead section-merge chain must not remain reachable from the Results";
}

TEST(PreComputeTest, RopeCacheTransformsMRopeWithTwoSections) {
    expect_mrope_cached(/*num_sections=*/2, /*section_width=*/3);
}

TEST(PreComputeTest, RopeCacheTransformsMRopeWithThreeSections) {
    expect_mrope_cached(/*num_sections=*/3, /*section_width=*/2);
}

// Section count must not be hardcoded to 3 (or any other fixed value) - verify a 4th explicit
// section (on top of the implicit first one) is discovered and merged too.
TEST(PreComputeTest, RopeCacheTransformsMRopeWithFourSections) {
    expect_mrope_cached(/*num_sections=*/4, /*section_width=*/2);
}

// Two sections explicitly claiming an overlapping channel sub-range would silently corrupt the
// merged LUT (one of them would be dropped); the matcher must leave Sin/Cos untouched instead.
TEST(PreComputeTest, RopeCacheRejectsMRopeWithOverlappingSections) {
    auto model = make_mrope_model(/*num_sections=*/3, /*section_width=*/2);

    // Rewrite the last section's Slice range to overlap the previous one's: [2,4) -> [1,3).
    const auto& ops = model->get_ops();
    for (const auto& op : ops) {
        auto slice = ov::as_type_ptr<ov::op::v8::Slice>(op);
        if (!slice) {
            continue;
        }
        auto begin = ov::as_type_ptr<ov::op::v0::Constant>(slice->input_value(1).get_node_shared_ptr());
        if (begin && begin->cast_vector<int64_t>() == std::vector<int64_t>{4}) {
            slice->input(1).replace_source_output(
                ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1})->output(0));
            slice->input(2).replace_source_output(
                ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {3})->output(0));
        }
    }

    ov::npuw::patterns::pre_compute::RopeCache pass(/*max_prompt_len=*/16);
    ASSERT_NO_THROW(pass.run_on_model(model));

    EXPECT_EQ(count_ops_of_type<ov::op::v0::Sin>(model), 1u) << "Sin/Cos must stay as-is, not be wrongly merged";
    EXPECT_EQ(count_ops_of_type<ov::op::v0::Cos>(model), 1u) << "Sin/Cos must stay as-is, not be wrongly merged";
}

// A channel range left uncovered by every explicit section doesn't need to form a contiguous
// (or even regular) run: those channels simply keep the implicit/base row's own value, exactly
// as the traced model's un-touched Gather("select") output would - so the match must still
// succeed instead of being rejected.
TEST(PreComputeTest, RopeCacheTransformsMRopeWithNonContiguousLeftoverChannels) {
    auto model = make_mrope_model(/*num_sections=*/3, /*section_width=*/2);

    // Shift the first explicit section's range (same width, so shapes stay valid) from [2,4) to
    // [1,3): still disjoint from the other explicit section ([4,6)), but now leaves behind two
    // separate uncovered channels ({0} and {3}) instead of one contiguous run.
    const auto& ops = model->get_ops();
    for (const auto& op : ops) {
        auto slice = ov::as_type_ptr<ov::op::v8::Slice>(op);
        if (!slice) {
            continue;
        }
        auto begin = ov::as_type_ptr<ov::op::v0::Constant>(slice->input_value(1).get_node_shared_ptr());
        if (begin && begin->cast_vector<int64_t>() == std::vector<int64_t>{2}) {
            slice->input(1).replace_source_output(
                ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1})->output(0));
            slice->input(2).replace_source_output(
                ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {3})->output(0));
        }
    }

    ov::npuw::patterns::pre_compute::RopeCache pass(/*max_prompt_len=*/16);
    ASSERT_NO_THROW(pass.run_on_model(model));

    EXPECT_EQ(count_ops_of_type<ov::op::v0::Sin>(model), 0u) << "Sin should be replaced by the merged LUT cache";
    EXPECT_EQ(count_ops_of_type<ov::op::v0::Cos>(model), 0u) << "Cos should be replaced by the merged LUT cache";
}

// Real-world models (e.g. Qwen3-VL-style "interleaved mrope") assign sections to channels with a
// stride > 1 instead of contiguous blocks - a layout a sorted-Slice-then-Concat reassembly of
// ranges cannot represent (only a ScatterUpdate-based merge can). This binds concrete position
// ids, constant-folds the merge graph, and checks the actual produced sin/cos values against a
// manually computed reference, so a channel-permutation bug (not just a crash) would be caught.
TEST(PreComputeTest, RopeCacheTransformsInterleavedMRopeSections) {
    const int64_t num_sections = 3;
    const int64_t full_width = 7;  // not evenly divisible by num_sections: counts become 3,2,2
    const int64_t seq = 4;

    auto model = make_mrope_model_interleaved(num_sections, full_width);

    ov::npuw::patterns::pre_compute::RopeCache pass(/*max_prompt_len=*/64);
    ASSERT_NO_THROW(pass.run_on_model(model));

    EXPECT_EQ(count_ops_of_type<ov::op::v0::Sin>(model), 0u) << "Sin should be replaced by the merged LUT cache";
    EXPECT_EQ(count_ops_of_type<ov::op::v0::Cos>(model), 0u) << "Cos should be replaced by the merged LUT cache";
    EXPECT_EQ(count_ops_of_type<ov::op::v3::ScatterNDUpdate>(model), 0u)
        << "the now-dead section-merge chain must not remain reachable from the Results";

    // Bind concrete position ids per rotary axis (row) and constant-fold the merge graph so the
    // actual produced values can be checked, not just "it doesn't crash". position_ids[row][0][s]
    // = 10*(row+1) + s - distinct per row so a channel accidentally reading the wrong row's
    // position id would be caught.
    std::vector<int64_t> position_ids_values;
    for (int64_t row = 0; row < num_sections; ++row) {
        for (int64_t s = 0; s < seq; ++s) {
            position_ids_values.push_back(10 * (row + 1) + s);
        }
    }
    auto position_ids_param = model->get_parameters().at(0);
    auto position_ids_const =
        ov::op::v0::Constant::create(ov::element::i64, position_ids_param->get_shape(), position_ids_values);
    ov::replace_node(position_ids_param, position_ids_const);

    ov::pass::Manager manager;
    manager.register_pass<ov::pass::ConstantFolding>();
    manager.run_passes(model);

    const auto& results = model->get_results();
    ASSERT_EQ(results.size(), 2u);
    auto sin_const = ov::as_type_ptr<ov::op::v0::Constant>(results[0]->input_value(0).get_node_shared_ptr());
    auto cos_const = ov::as_type_ptr<ov::op::v0::Constant>(results[1]->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(sin_const) << "the merge graph must fully constant-fold once position_ids is concrete";
    ASSERT_TRUE(cos_const);

    const auto sin_values = sin_const->cast_vector<float>();
    const auto cos_values = cos_const->cast_vector<float>();
    ASSERT_EQ(sin_values.size(), static_cast<size_t>(seq * full_width * 2));

    for (int64_t s = 0; s < seq; ++s) {
        for (int64_t c = 0; c < full_width; ++c) {
            // Interleaved assignment used by make_mrope_model_interleaved(): channel c belongs to
            // rotary axis row (c % num_sections).
            const int64_t row = c % num_sections;
            const float position = static_cast<float>(10 * (row + 1) + s);
            const float inv_freq = 0.1f + 0.05f * static_cast<float>(c);
            const float angle = position * inv_freq;
            for (int64_t half : {0, 1}) {  // torch.cat([row, row], dim=-1) duplication
                const size_t idx = static_cast<size_t>(s * (full_width * 2) + half * full_width + c);
                EXPECT_NEAR(sin_values[idx], std::sin(angle), 1e-2f) << "s=" << s << " c=" << c << " half=" << half;
                EXPECT_NEAR(cos_values[idx], std::cos(angle), 1e-2f) << "s=" << s << " c=" << c << " half=" << half;
            }
        }
    }
}

}  // namespace
