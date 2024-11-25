// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <openvino/opsets/opset13.hpp>
#include <transformations/cpu_opset/common/op/paged_attention_split.hpp>
#include <transformations/cpu_opset/common/pass/paged_attention_split_fusion.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <transformations/utils/gen_pattern.hpp>
#include <openvino/op/paged_attention.hpp>
#include <openvino/pass/manager.hpp>
#include <ov_ops/rotary_positional_embeddings.hpp>
#include <ov_ops/type_relaxed.hpp>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/partial_shape.hpp"
#include "transformations/utils/print_model.hpp"

using namespace testing;
using namespace ov;
using namespace ov::intel_cpu;
using namespace ov::gen_pattern;

namespace {
    const auto q_head_num = 32;
    const auto kv_head_num = 2;
    const auto head_size = 128;
    auto q_hidden_size = q_head_num * head_size;
    auto kv_hidden_size = kv_head_num * head_size;
    const auto out_hidden_size = q_head_num * head_size;
    const auto rotary_size = 64;
};

static std::shared_ptr<ov::Model> makePASubgraph() {
    auto fc_output = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{1, -1, (q_head_num + 2 * kv_head_num) * head_size});
    auto cos_sin = std::make_shared<ov::op::v0::Parameter>(element::f32, ov::PartialShape{1, -1, rotary_size / 2, 2});
    auto kv_cache = std::make_shared<ov::op::v0::Parameter>(element::f32, ov::PartialShape{-1, kv_head_num, 32, head_size});
    auto dummy_input = std::make_shared<ov::op::v0::Parameter>(element::i32, ov::PartialShape{-1});
    auto scale = makeConst(element::f32, {}, {0});
    auto sliding = makeConst(element::i32, {}, {0});
    auto alibi = makeConst(element::f32, {1}, {0});
    auto max_context_len = makeConst(element::i32, {}, {0});

    auto aten_cat_Concat = makeOP<ov::op::internal::RoPE>({fc_output, cos_sin, cos_sin},
        {
            {"config.slice_start", 0},
            {"config.slice_stop", q_hidden_size},
            {"config.input_trans0213", false},
            {"config.is_interleaved", false},
            {"config.rotary_ndims", rotary_size},
            {"config.is_chatglm", true},
            {"config.support_2d_rope", false},
            {"config.is_qwen", false},
            {"config.head_cnt", q_head_num},
            {"config.head_size", head_size},
            {"config.gather_position_arg_id", 0}
        }
    );
    auto Transpose_19027 = makeOP<opset1::Reshape>({aten_cat_Concat, {-1, 1, 0, 0}}, {{"special_zero", true}});
    auto Reshape_19029 = makeOP<opset1::Reshape>({Transpose_19027, {0, -1}}, {{"special_zero", true}});
    auto aten_cat_Concat_1 = makeOP<ov::op::internal::RoPE>({fc_output, cos_sin, cos_sin},
        {
            {"config.slice_start", q_hidden_size},
            {"config.slice_stop", q_hidden_size + kv_hidden_size},
            {"config.input_trans0213", false},
            {"config.is_interleaved", false},
            {"config.rotary_ndims", rotary_size},
            {"config.is_chatglm", true},
            {"config.support_2d_rope", false},
            {"config.is_qwen", false},
            {"config.head_cnt", kv_head_num},
            {"config.head_size", head_size},
            {"config.gather_position_arg_id", 0}
        }
    );
    auto Transpose_19032 = makeOP<opset1::Reshape>({aten_cat_Concat_1, {-1, 1, 0, 0}}, {{"special_zero", true}});
    auto Reshape_19039 = makeOP<opset1::Reshape>({Transpose_19032, {0, -1}}, {{"special_zero", true}});
    auto prim_ListUnpack = makeOP<opset1::VariadicSplit>({fc_output, -1, {q_hidden_size, kv_hidden_size, kv_hidden_size}});
    auto aten_view_Reshape_3 = makeOP<opset1::Reshape>({prim_ListUnpack->output(2), {0, 0, kv_hidden_size / head_size, head_size}},
        {{"special_zero", true}});
    auto Transpose_19036 = makeOP<opset1::Reshape>({aten_view_Reshape_3, {-1, 1, 0, 0}}, {{"special_zero", true}});
    auto Reshape_19041 = makeOP<opset1::Reshape>({Transpose_19036, {0,-1}}, {{"special_zero", true}});
    
    auto PagedAttentionExtension_19048 = std::make_shared<ov::op::PagedAttentionExtension>(OutputVector{Reshape_19029, Reshape_19039, Reshape_19041,
        kv_cache, kv_cache, dummy_input, dummy_input, dummy_input, dummy_input, scale, sliding, alibi, max_context_len});
    auto Reshape_19059 = makeOP<opset1::Reshape>({PagedAttentionExtension_19048->output(0), {0, 1, -1, head_size}}, {{"special_zero", true}});
    auto core_attention_aten_permute_Transpose_3 = makeOP<opset1::Reshape>({Reshape_19059, {1, -1, 0, 0}}, {{"special_zero", true}});
    auto core_attention_aten_reshape_Reshape = makeOP<opset1::Reshape>({core_attention_aten_permute_Transpose_3, {0, 0, out_hidden_size}},
        {{"special_zero", true}});

    ResultVector results{std::make_shared<ov::op::v0::Result>(core_attention_aten_reshape_Reshape)};
    return std::make_shared<Model>(results, ParameterVector{fc_output, cos_sin, kv_cache, dummy_input}, "PASplit");
}

TEST(TransformationTests, PagedAttentionWithSplitFusion) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        using namespace ov;
        {
            f = makePASubgraph();
            pass::Manager m;
            m.register_pass<ov::pass::InitNodeInfo>();
            m.register_pass<PagedAttentionFusion>();
            m.run_passes(f);
        }
        //construct ref interaction
        {
            auto fc_output = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{1, -1, (q_head_num + 2 * kv_head_num) * head_size});
            auto cos_sin = std::make_shared<ov::op::v0::Parameter>(element::f32, ov::PartialShape{1, -1, rotary_size / 2, 2});
            auto kv_cache = std::make_shared<ov::op::v0::Parameter>(element::f32, ov::PartialShape{-1, kv_head_num, 32, head_size});
            auto dummy_input = std::make_shared<ov::op::v0::Parameter>(element::i32, ov::PartialShape{-1});
            auto scale = makeConst(element::f32, {}, {0});
            auto sliding = makeConst(element::i32, {}, {0});
            auto alibi = makeConst(element::f32, {1}, {0});
            auto max_context_len = makeConst(element::i32, {}, {0});

            auto rope_q = makeOP<ov::op::internal::RoPE>({fc_output, cos_sin, cos_sin},
                {
                    {"config.slice_start", 0},
                    {"config.slice_stop", q_hidden_size},
                    {"config.input_trans0213", false},
                    {"config.is_interleaved", false},
                    {"config.rotary_ndims", rotary_size},
                    {"config.is_chatglm", true},
                    {"config.support_2d_rope", false},
                    {"config.is_qwen", false},
                    {"config.head_cnt", q_head_num},
                    {"config.head_size", head_size},
                    {"config.gather_position_arg_id", 0}
                }
            );
            auto rope_k = makeOP<ov::op::internal::RoPE>({fc_output, cos_sin, cos_sin},
                {
                    {"config.slice_start", q_hidden_size},
                    {"config.slice_stop", q_hidden_size + kv_hidden_size},
                    {"config.input_trans0213", false},
                    {"config.is_interleaved", false},
                    {"config.rotary_ndims", rotary_size},
                    {"config.is_chatglm", true},
                    {"config.support_2d_rope", false},
                    {"config.is_qwen", false},
                    {"config.head_cnt", kv_head_num},
                    {"config.head_size", head_size},
                    {"config.gather_position_arg_id", 0}
                }
            );

            Extensions::Cpu::PagedAttentionFuseConfig config;
            config.fuse_reshape_split = true;
            config.is_seq_len_first = true;
            config.slice_start = static_cast<size_t>(q_hidden_size + kv_hidden_size);
            config.slice_stop = config.slice_start + static_cast<size_t>(kv_hidden_size);
            config.v_head_size = static_cast<size_t>(head_size);
            config.out_hidden_size = static_cast<size_t>(out_hidden_size);
            config.output_type[0] = element::f32;
            config.output_type[1] = element::f32;
            ov::OutputVector args = {
                rope_q, rope_k, fc_output, kv_cache, kv_cache,
                dummy_input, dummy_input, dummy_input, dummy_input, scale, sliding, alibi, max_context_len
            };
            auto pa_node = std::make_shared<ov::intel_cpu::PagedAttentionWithSplit>(args, config);
            ResultVector results{std::make_shared<ov::op::v0::Result>(pa_node->output(0))};
            f_ref = std::make_shared<Model>(results, ParameterVector{fc_output, cos_sin, kv_cache, dummy_input}, "PASplitRef");
        }
        auto res = compare_functions(f, f_ref);
        ASSERT_TRUE(res.first) << res.second;
    }
}
