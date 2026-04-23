// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "embedding/prepare_embedding_model.hpp"
#include "util.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"

namespace {

std::shared_ptr<ov::Model> build_embedding_like_sdpa_model(const std::string& sdpa_name) {
    using namespace ov;

    auto input_ids = std::make_shared<op::v0::Parameter>(element::i64, Shape{1, 1});
    input_ids->set_friendly_name("input_ids");
    input_ids->get_output_tensor(0).set_names({"input_ids"});

    auto attention_mask = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 1});
    attention_mask->set_friendly_name("attention_mask");
    attention_mask->get_output_tensor(0).set_names({"attention_mask"});

    auto q = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 1, 1, 4});
    auto k_a = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 1, 1, 4});
    auto k_b = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 1, 1, 4});
    auto v_in = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 1, 4, 1});
    auto rope_matmul_input = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 1, 1});

    auto attn_mask_unsqueeze_axis = op::v0::Constant::create(element::i64, Shape{2}, {1, 2});
    auto attn_mask_unsqueezed = std::make_shared<op::v0::Unsqueeze>(attention_mask, attn_mask_unsqueeze_axis);

    auto unsqueeze_axis = op::v0::Constant::create(element::i64, Shape{1}, {2});

    auto c1 = op::v0::Constant::create(element::i64, Shape{1}, {1});
    auto c4 = op::v0::Constant::create(element::i64, Shape{1}, {4});
    auto shape_concat = std::make_shared<op::v0::Concat>(OutputVector{c1, c1, c1, c1, c4}, 0);

    auto reshape_target = op::v0::Constant::create(element::i64, Shape{4}, {1, 1, 1, 4});
    auto transpose_order = op::v0::Constant::create(element::i64, Shape{4}, {0, 1, 3, 2});

    auto k_add = std::make_shared<op::v1::Add>(k_a, k_b);
    auto k_unsqueeze = std::make_shared<op::v0::Unsqueeze>(k_add, unsqueeze_axis);
    auto k_broadcast = std::make_shared<op::v3::Broadcast>(k_unsqueeze, shape_concat);
    auto k_reshape = std::make_shared<op::v1::Reshape>(k_broadcast, reshape_target, false);

    auto v_transpose = std::make_shared<op::v1::Transpose>(v_in, transpose_order);
    auto v_unsqueeze = std::make_shared<op::v0::Unsqueeze>(v_transpose, unsqueeze_axis);
    auto v_broadcast = std::make_shared<op::v3::Broadcast>(v_unsqueeze, shape_concat);
    auto v_reshape = std::make_shared<op::v1::Reshape>(v_broadcast, reshape_target, false);

    auto scale = op::v0::Constant::create(element::f32, Shape{1}, {1.0f});
    auto sdpa =
        std::make_shared<op::v13::ScaledDotProductAttention>(q, k_reshape, v_reshape, attn_mask_unsqueezed, scale, false);
    sdpa->set_friendly_name(sdpa_name);

    auto result = std::make_shared<op::v0::Result>(sdpa);

    auto zero_i = op::v0::Constant::create(element::i64, Shape{}, {0});
    auto one_i = op::v0::Constant::create(element::i64, Shape{}, {1});
    auto range = std::make_shared<op::v4::Range>(zero_i, one_i, one_i, element::i64);

    auto unsqueeze_axis0 = op::v0::Constant::create(element::i64, Shape{1}, {0});
    auto unsqueeze = std::make_shared<op::v0::Unsqueeze>(range, unsqueeze_axis0);
    auto unsqueeze1 = std::make_shared<op::v0::Unsqueeze>(unsqueeze, unsqueeze_axis0);
    auto convert = std::make_shared<op::v0::Convert>(unsqueeze1, element::f32);

    auto matmul = std::make_shared<op::v0::MatMul>(rope_matmul_input, convert, false, false);
    auto rope_transpose_order = op::v0::Constant::create(element::i64, Shape{3}, {0, 2, 1});
    auto transpose = std::make_shared<op::v1::Transpose>(matmul, rope_transpose_order);
    auto rope_concat = std::make_shared<op::v0::Concat>(OutputVector{transpose, transpose}, 2);
    auto cos = std::make_shared<op::v0::Cos>(rope_concat);
    auto sin = std::make_shared<op::v0::Sin>(rope_concat);
    auto cos_result = std::make_shared<op::v0::Result>(cos);
    auto sin_result = std::make_shared<op::v0::Result>(sin);

    return std::make_shared<Model>(ResultVector{result, cos_result, sin_result},
                                   ParameterVector{input_ids, attention_mask, q, k_a, k_b, v_in, rope_matmul_input});
}

TEST(PrepareEmbeddingModelTest, ThrowsOnLayerIdOverflowInSdpaName) {
    auto model = build_embedding_like_sdpa_model("layers.999999999999999999999999999.self_attn");

    ov::npuw::util::PrepareTextEmbeddingModel pass(/*seq_len_dim=*/2);

    EXPECT_ANY_THROW(pass.run_on_model(model));
}

TEST(PrepareEmbeddingModelTest, IgnoresNonNumericLayerIdInSdpaName) {
    auto model = build_embedding_like_sdpa_model("layers.string.self_attn");

    ov::npuw::util::PrepareTextEmbeddingModel pass(/*seq_len_dim=*/2);

    EXPECT_NO_THROW(pass.run_on_model(model));
    EXPECT_ANY_THROW(model->input(ov::npuw::util::make_past_key_name(12)));
    EXPECT_ANY_THROW(model->input(ov::npuw::util::make_past_value_name(12)));
    EXPECT_ANY_THROW(model->output(ov::npuw::util::make_present_key_name(12)));
    EXPECT_ANY_THROW(model->output(ov::npuw::util::make_present_value_name(12)));
}

TEST(PrepareEmbeddingModelTest, AcceptsValidLayerIdInSdpaName) {
    auto model = build_embedding_like_sdpa_model("layers.12.self_attn");

    ov::npuw::util::PrepareTextEmbeddingModel pass(/*seq_len_dim=*/2);

    try {
        pass.run_on_model(model);
    } catch (const std::exception&) {
        // Name generation happens before late model validation failures in this synthetic graph.
    }
    EXPECT_NO_THROW(model->input(ov::npuw::util::make_past_key_name(12)));
    EXPECT_NO_THROW(model->input(ov::npuw::util::make_past_value_name(12)));
    EXPECT_NO_THROW(model->output(ov::npuw::util::make_present_key_name(12)));
    EXPECT_NO_THROW(model->output(ov::npuw::util::make_present_value_name(12)));
}

}  // namespace
