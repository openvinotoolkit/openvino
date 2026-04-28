// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/op/ops.hpp"
#include "openvino/pass/manager.hpp"
#include "ov_ops/rotary_positional_embeddings.hpp"
#include "partitioning/patterns/pre_compute.hpp"
#include "transformations/common_optimizations/fuse_rotary_positional_embeddings.hpp"

namespace {

std::shared_ptr<ov::Model> build_llama2_like_rope_graph() {
    using namespace ov;

    constexpr int64_t kBatch = 1;
    constexpr int64_t kHeads = 1;
    constexpr int64_t kSeqLen = 16;
    constexpr int64_t kRotaryNdims = 128;
    constexpr int64_t kHalfNdims = kRotaryNdims / 2;

    auto x = std::make_shared<op::v0::Parameter>(element::f32, Shape{kBatch, kHeads, kSeqLen, kRotaryNdims});
    auto position_ids = std::make_shared<op::v0::Parameter>(element::i64, Shape{kBatch, kHeads, kSeqLen});

    auto inv_freq_data = std::vector<float>(kHalfNdims, 1.0f);
    auto inv_freq = op::v0::Constant::create(element::f32, Shape{kBatch, kHeads, kHalfNdims, 1}, inv_freq_data);

    auto shape_of = std::make_shared<op::v3::ShapeOf>(x, element::i64);
    auto gather_idx = op::v0::Constant::create(element::i64, Shape{2}, {0, 1});
    auto gather_axis = op::v0::Constant::create(element::i64, Shape{}, {0});
    auto gather = std::make_shared<op::v8::Gather>(shape_of, gather_idx, gather_axis);

    auto shape_half = op::v0::Constant::create(element::i64, Shape{1}, {kHalfNdims});
    auto shape_seq = op::v0::Constant::create(element::i64, Shape{1}, {kSeqLen});
    auto concat_1 = std::make_shared<op::v0::Concat>(OutputVector{gather, shape_half, shape_seq}, 0);

    auto broadcast = std::make_shared<op::v3::Broadcast>(inv_freq, concat_1);

    auto unsqueeze_axis = op::v0::Constant::create(element::i64, Shape{1}, {-1});
    auto pos_unsqueeze = std::make_shared<op::v0::Unsqueeze>(position_ids, unsqueeze_axis);
    auto pos_convert = std::make_shared<op::v0::Convert>(pos_unsqueeze, element::f32);

    auto matmul = std::make_shared<op::v0::MatMul>(broadcast, pos_convert, false, false);
    auto transpose_axes = op::v0::Constant::create(element::i64, Shape{4}, {0, 1, 3, 2});
    auto transpose = std::make_shared<op::v1::Transpose>(matmul, transpose_axes);

    auto concat_2 = std::make_shared<op::v0::Concat>(OutputVector{transpose, transpose}, -1);
    auto cos = std::make_shared<op::v0::Cos>(concat_2);
    auto sin = std::make_shared<op::v0::Sin>(concat_2);

    auto split_lengths = op::v0::Constant::create(element::i64, Shape{2}, {kHalfNdims, kHalfNdims});
    auto split_axis = op::v0::Constant::create(element::i64, Shape{}, {3});
    auto split = std::make_shared<op::v1::VariadicSplit>(x, split_axis, split_lengths);
    split->set_output_size(2);

    auto minus_one = op::v0::Constant::create(element::f32, Shape{1}, {-1.0f});
    auto neg_half = std::make_shared<op::v1::Multiply>(split->output(1), minus_one);
    auto rotate_half = std::make_shared<op::v0::Concat>(OutputVector{neg_half, split->output(0)}, -1);

    auto mul_cos = std::make_shared<op::v1::Multiply>(x, cos);
    auto mul_sin = std::make_shared<op::v1::Multiply>(rotate_half, sin);
    auto rope_formula = std::make_shared<op::v1::Add>(mul_cos, mul_sin);

    auto result = std::make_shared<op::v0::Result>(rope_formula);
    return std::make_shared<Model>(ResultVector{result}, ParameterVector{x, position_ids});
}

size_t count_internal_rope_nodes(const std::shared_ptr<ov::Model>& model) {
    size_t count = 0;
    for (const auto& node : model->get_ops()) {
        if (std::dynamic_pointer_cast<ov::op::internal::RoPE>(node)) {
            count++;
        }
    }
    return count;
}

TEST(NPUW_RopePrecomputeAndFusion, Llama2LikePatternCreatesInternalRope) {
    auto model = build_llama2_like_rope_graph();

    EXPECT_EQ(count_internal_rope_nodes(model), 0);

    ov::pass::Manager manager;
    manager.register_pass<ov::npuw::patterns::pre_compute::RopeCache>(2048, "");
    manager.register_pass<ov::pass::RoPEFusion>();

    manager.run_passes(model);
    model->validate_nodes_and_infer_types();

    EXPECT_GE(count_internal_rope_nodes(model), 1);
}

}  // namespace
