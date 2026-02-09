// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <string>

#include "model_builder/model_builder.hpp"
#include "openvino/openvino.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/pass/sdpa_to_paged_attention.hpp"
#include "openvino/pass/stateful_to_stateless.hpp"

namespace {

using ov::test::npuw::LLMConfig;
using ov::test::npuw::ModelBuilder;

LLMConfig make_small_config() {
    LLMConfig config;
    config.hidden_size = 16;
    config.num_heads = 2;
    config.head_dim = 8;
    config.intermediate_size = 32;
    config.vocab_size = 32;
    config.num_layers = 1;
    config.context_len = 8;
    config.seq_len = 2;
    config.precision = ov::element::f32;
    return config;
}

template <typename T>
size_t count_ops(const std::shared_ptr<ov::Model>& model) {
    size_t count = 0;
    for (const auto& node : model->get_ordered_ops()) {
        if (ov::is_type<T>(node)) {
            ++count;
        }
    }
    return count;
}

bool has_op_type_name(const std::shared_ptr<ov::Model>& model, const std::string& type_name) {
    for (const auto& node : model->get_ordered_ops()) {
        if (node->get_type_name() == type_name) {
            return true;
        }
    }
    return false;
}

std::shared_ptr<ov::Node> find_by_name(const std::shared_ptr<ov::Model>& model, const std::string& name) {
    for (const auto& node : model->get_ordered_ops()) {
        if (node->get_friendly_name() == name) {
            return node;
        }
    }
    return nullptr;
}

}  // namespace

TEST(LLMBuilderTest, KVCacheCreatesStateOpsAndSinks) {
    auto config = make_small_config();
    config.use_kv_cache = true;
    config.use_position_ids = false;

    ModelBuilder builder;
    auto model = builder.build_llm(config);

    EXPECT_EQ(count_ops<ov::op::v6::ReadValue>(model), 2);
    EXPECT_EQ(count_ops<ov::op::v6::Assign>(model), 2);
    EXPECT_EQ(model->get_sinks().size(), 2);
    EXPECT_GE(count_ops<ov::op::v13::ScaledDotProductAttention>(model), 1);
}

TEST(LLMBuilderTest, StatefulToStatelessRemovesStateOps) {
    auto config = make_small_config();
    config.use_kv_cache = true;
    config.use_position_ids = false;

    ModelBuilder builder;
    auto model = builder.build_llm(config);

    ov::pass::StatefulToStateless pass;
    EXPECT_NO_THROW(pass.run_on_model(model));

    EXPECT_EQ(count_ops<ov::op::v6::ReadValue>(model), 0);
    EXPECT_EQ(count_ops<ov::op::v6::Assign>(model), 0);
    EXPECT_EQ(model->get_sinks().size(), 0);
}

TEST(LLMBuilderTest, SDPAToPagedAttentionTransforms) {
    auto config = make_small_config();
    config.use_kv_cache = true;
    config.use_position_ids = true;

    ModelBuilder builder;
    auto model = builder.build_llm(config);

    ov::pass::SDPAToPagedAttention pass;
    EXPECT_NO_THROW(pass.run_on_model(model));

    EXPECT_TRUE(has_op_type_name(model, "PagedAttentionExtension"));
}

TEST(LLMBuilderTest, RoPEToggleAddsAndRemovesSubgraph) {
    auto config = make_small_config();
    config.use_kv_cache = false;
    config.use_position_ids = true;

    ModelBuilder builder;
    auto model_with_rope = builder.build_llm(config);

    EXPECT_NE(find_by_name(model_with_rope, "model.layers.0.q_rope"), nullptr);
    EXPECT_NE(find_by_name(model_with_rope, "model.layers.0.k_rope"), nullptr);

    config.use_position_ids = false;
    auto model_without_rope = builder.build_llm(config);

    EXPECT_EQ(find_by_name(model_without_rope, "model.layers.0.q_rope"), nullptr);
    EXPECT_EQ(find_by_name(model_without_rope, "model.layers.0.k_rope"), nullptr);
}

TEST(LLMBuilderTest, GQAToggleAddsAndRemovesRepeatKV) {
    auto config = make_small_config();
    config.use_kv_cache = false;
    config.num_heads = 4;
    config.num_kv_heads = 2;

    ModelBuilder builder;
    auto model_with_gqa = builder.build_llm(config);

    EXPECT_NE(find_by_name(model_with_gqa, "model.layers.0.k_repeat"), nullptr);
    EXPECT_NE(find_by_name(model_with_gqa, "model.layers.0.v_repeat"), nullptr);

    config.num_kv_heads = 0;
    auto model_without_gqa = builder.build_llm(config);

    EXPECT_EQ(find_by_name(model_without_gqa, "model.layers.0.k_repeat"), nullptr);
    EXPECT_EQ(find_by_name(model_without_gqa, "model.layers.0.v_repeat"), nullptr);
}

TEST(LLMBuilderTest, AttentionMaskIs4DAndBroadcastable) {
    auto config = make_small_config();
    config.use_kv_cache = false;
    config.use_position_ids = false;

    ModelBuilder builder;
    auto model = builder.build_llm(config);

    auto mask = find_by_name(model, "model.mask_4d");
    ASSERT_NE(mask, nullptr);

    const auto shape = mask->get_output_partial_shape(0);
    ASSERT_TRUE(shape.rank().is_static());
    EXPECT_EQ(shape.rank().get_length(), 4);
    EXPECT_TRUE(shape[1].is_static());
    EXPECT_EQ(shape[1].get_length(), 1);
}

TEST(LLMBuilderTest, CpuInferenceSmokeTest) {
    ov::Core core;
    auto devices = core.get_available_devices();
    if (std::find(devices.begin(), devices.end(), "CPU") == devices.end()) {
        GTEST_SKIP() << "CPU device is not available.";
    }

    auto config = make_small_config();
    config.use_kv_cache = false;
    config.use_position_ids = false;

    ModelBuilder builder;
    auto model = builder.build_llm(config);

    auto compiled = core.compile_model(model, "CPU");
    auto request = compiled.create_infer_request();

    ov::Tensor input_ids(ov::element::i64, ov::Shape{1, 2});
    ov::Tensor attention_mask(ov::element::i64, ov::Shape{1, 2});
    ov::Tensor beam_idx(ov::element::i32, ov::Shape{1});

    auto* ids_data = input_ids.data<int64_t>();
    ids_data[0] = 1;
    ids_data[1] = 2;

    auto* mask_data = attention_mask.data<int64_t>();
    mask_data[0] = 1;
    mask_data[1] = 1;

    beam_idx.data<int32_t>()[0] = 0;

    request.set_tensor("input_ids", input_ids);
    request.set_tensor("attention_mask", attention_mask);
    request.set_tensor("beam_idx", beam_idx);

    EXPECT_NO_THROW(request.infer());

    auto output = request.get_output_tensor(0);
    EXPECT_EQ(output.get_element_type(), ov::element::f32);
    EXPECT_EQ(output.get_shape(), (ov::Shape{1, 2, config.vocab_size}));
}
