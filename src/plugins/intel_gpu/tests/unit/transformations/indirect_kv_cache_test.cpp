// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "common_test_utils/graph_comparator.hpp"
#include "common_test_utils/ov_test_utils.hpp"

#include <string>
#include <memory>

#include "openvino/core/model.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/manager.hpp"

#include <transformations/utils/utils.hpp>
#include "plugin/transformations/indirect_kv_cache.hpp"

#include "intel_gpu/op/indirect_gemm.hpp"
#include "intel_gpu/op/gemm.hpp"
#include "intel_gpu/op/indirect_sdpa.hpp"
#include "intel_gpu/op/sdpa.hpp"
#include "intel_gpu/op/read_value.hpp"
#include "intel_gpu/op/kv_cache.hpp"
#include "intel_gpu/op/sparse_sdpa.hpp"

using namespace testing;
using namespace ov::intel_gpu;

TEST_F(TransformationTestsF, IndirectKVCache1) {
    std::vector<int64_t> in0_order = {0, 1, 2, 3};
    std::vector<int64_t> in1_order = {0, 1, 3, 2};
    std::vector<int64_t> out_order = {0, 1, 2, 3};
    {
        auto variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{1, 32, -1, 80}, ov::element::f32, "v0"});
        auto new_token_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 32, -1, 80});
        auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1});
        auto past = std::make_shared<ov::intel_gpu::op::ReadValue>(variable);
        auto axis = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, 0);
        auto gather_past = std::make_shared<ov::op::v8::Gather>(past, beam_idx, axis);
        auto kv_cache = std::make_shared<ov::intel_gpu::op::KVCache>(gather_past, new_token_param, variable, 2, ov::element::f32);
        auto gemm_in = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 32, -1, -1});
        auto gemm = std::make_shared<ov::intel_gpu::op::Gemm>(gemm_in, kv_cache, in0_order, in1_order, out_order);
        auto result = std::make_shared<ov::op::v0::Result>(gemm);

        model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{new_token_param, beam_idx, gemm_in});
        manager.register_pass<IndirectKVCache>();
    }
    {
        auto variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{1, 32, -1, 80}, ov::element::f32, "v0"});
        auto parameter = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 32, -1, 80});
        auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1});
        auto past = std::make_shared<ov::intel_gpu::op::ReadValue>(variable);
        auto kv_cache = std::make_shared<ov::intel_gpu::op::KVCache>(past, parameter, beam_idx, variable, 2, 0, ov::element::f32);
        auto gemm_in = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 32, -1, -1});
        auto gemm = std::make_shared<ov::intel_gpu::op::IndirectGemm>(gemm_in, kv_cache->output(0), kv_cache->output(1), false, true, 0,
                                                                      in0_order, in1_order, out_order);
        auto result = std::make_shared<ov::op::v0::Result>(gemm);

        model_ref = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{parameter, beam_idx, gemm_in});
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
}

TEST_F(TransformationTestsF, IndirectKVCache2) {
    std::vector<int64_t> in0_order = {0, 1, 2, 3};
    std::vector<int64_t> in1_order = {0, 1, 3, 2};
    std::vector<int64_t> out_order = {0, 1, 2, 3};
    {
        auto variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{1, 32, -1, 80}, ov::element::f32, "v0"});
        auto new_token_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 32, -1, 80});
        auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1});
        auto past = std::make_shared<ov::intel_gpu::op::ReadValue>(variable);
        auto axis = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, 0);
        auto gather_past = std::make_shared<ov::op::v8::Gather>(past, beam_idx, axis);
        auto kv_cache = std::make_shared<ov::intel_gpu::op::KVCache>(gather_past, new_token_param, variable, 2, ov::element::f32);
        auto gemm_in = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 32, -1, -1});
        auto gemm = std::make_shared<ov::intel_gpu::op::Gemm>(kv_cache, gemm_in, in0_order, in1_order, out_order);
        auto result = std::make_shared<ov::op::v0::Result>(gemm);

        model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{new_token_param, beam_idx, gemm_in});
        manager.register_pass<IndirectKVCache>();
    }
    {
        auto variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{1, 32, -1, 80}, ov::element::f32, "v0"});
        auto parameter = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 32, -1, 80});
        auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1});
        auto past = std::make_shared<ov::intel_gpu::op::ReadValue>(variable);
        auto kv_cache = std::make_shared<ov::intel_gpu::op::KVCache>(past, parameter, beam_idx, variable, 2, 0, ov::element::f32);
        auto gemm_in = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 32, -1, -1});
        auto gemm = std::make_shared<ov::intel_gpu::op::IndirectGemm>(kv_cache->output(0), gemm_in, kv_cache->output(1), true, false, 0,
                                                                      in0_order, in1_order, out_order);
        auto result = std::make_shared<ov::op::v0::Result>(gemm);

        model_ref = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{parameter, beam_idx, gemm_in});
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
}

TEST_F(TransformationTestsF, IndirectKVCache3) {
    std::vector<int64_t> in0_order = {0, 1, 2, 3};
    std::vector<int64_t> in1_order = {0, 1, 3, 2};
    std::vector<int64_t> out_order = {0, 1, 2, 3};
    {
        auto variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{1, 32, -1, 80}, ov::element::f32, "v0"});
        auto new_token_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 32, -1, 80});
        auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{32});
        auto past = std::make_shared<ov::intel_gpu::op::ReadValue>(variable);
        auto axis = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, 1);
        auto gather_past = std::make_shared<ov::op::v8::Gather>(past, beam_idx, axis);
        auto kv_cache = std::make_shared<ov::intel_gpu::op::KVCache>(gather_past, new_token_param, variable, 2, ov::element::f32);
        auto gemm_in = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 32, -1, -1});
        auto gemm = std::make_shared<ov::intel_gpu::op::Gemm>(gemm_in, kv_cache, in0_order, in1_order, out_order);
        auto result = std::make_shared<ov::op::v0::Result>(gemm);

        model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{new_token_param, beam_idx, gemm_in});
        manager.register_pass<IndirectKVCache>();
    }
    {
        auto variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{1, 32, -1, 80}, ov::element::f32, "v0"});
        auto parameter = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 32, -1, 80});
        auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{32});
        auto past = std::make_shared<ov::intel_gpu::op::ReadValue>(variable);
        auto kv_cache = std::make_shared<ov::intel_gpu::op::KVCache>(past, parameter, beam_idx, variable, 2, 1, ov::element::f32);
        auto gemm_in = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 32, -1, -1});
        auto gemm = std::make_shared<ov::intel_gpu::op::IndirectGemm>(gemm_in, kv_cache->output(0), kv_cache->output(1), false, true, 1,
                                                                      in0_order, in1_order, out_order);
        auto result = std::make_shared<ov::op::v0::Result>(gemm);

        model_ref = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{parameter, beam_idx, gemm_in});
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
}

TEST_F(TransformationTestsF, IndirectKVCache4) {
    std::vector<int64_t> in0_order = {0, 1, 2, 3};
    std::vector<int64_t> in1_order = {0, 2, 1, 3};
    std::vector<int64_t> in2_order = {0, 2, 1, 3};
    std::vector<int64_t> out_order = {0, 1, 2, 3};
    const bool is_causal = true;
    {
        auto key_variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{1, -1, 32, 128}, ov::element::f32, "v0"});
        auto value_variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{1, -1, 32, 128}, ov::element::f32, "v1"});
        auto parameter_key = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, 32, 128});
        auto parameter_value = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, 32, 128});
        auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1});
        auto key_past = std::make_shared<ov::intel_gpu::op::ReadValue>(key_variable);
        auto value_past = std::make_shared<ov::intel_gpu::op::ReadValue>(value_variable);
        auto axis = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, 1);
        auto key_gather_past = std::make_shared<ov::op::v8::Gather>(key_past, beam_idx, axis);
        auto value_gather_past = std::make_shared<ov::op::v8::Gather>(value_past, beam_idx, axis);
        auto key_cache = std::make_shared<ov::intel_gpu::op::KVCache>(key_gather_past, parameter_key, key_variable, 2, ov::element::f32);
        auto value_cache = std::make_shared<ov::intel_gpu::op::KVCache>(value_gather_past, parameter_value, value_variable, 2, ov::element::f32);
        auto sdpa_q = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 32, -1, 128});
        auto inputs = ov::OutputVector{sdpa_q, key_cache, value_cache};
        auto sdpa = std::make_shared<ov::intel_gpu::op::SDPA>(inputs, is_causal, in0_order, in1_order, in2_order, out_order);
        auto result = std::make_shared<ov::op::v0::Result>(sdpa);

        model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{parameter_key, parameter_value, beam_idx, sdpa_q});
        manager.register_pass<IndirectKVCache>();
    }
    {
        auto indirect_axis = 1;
        auto key_variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{1, -1, 32, 128}, ov::element::f32, "v0"});
        auto value_variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{1, -1, 32, 128}, ov::element::f32, "v1"});
        auto parameter_key = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, 32, 128});
        auto parameter_value = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, 32, 128});
        auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1});
        auto key_past = std::make_shared<ov::intel_gpu::op::ReadValue>(key_variable);
        auto value_past = std::make_shared<ov::intel_gpu::op::ReadValue>(value_variable);
        auto key_cache = std::make_shared<ov::intel_gpu::op::KVCache>(key_past, parameter_key, beam_idx, key_variable, 2, indirect_axis, ov::element::f32);
        auto value_cache = std::make_shared<ov::intel_gpu::op::KVCache>(value_past, parameter_value, beam_idx, value_variable, 2, indirect_axis, ov::element::f32);
        auto sdpa_q = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 32, -1, 128});
        auto inputs = ov::OutputVector{sdpa_q, key_cache, value_cache};
        auto sdpa = std::make_shared<ov::intel_gpu::op::IndirectSDPA>(inputs, key_cache->output(1), is_causal, indirect_axis, in0_order, in1_order, in2_order, out_order);
        auto result = std::make_shared<ov::op::v0::Result>(sdpa);

        model_ref = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{parameter_key, parameter_value, beam_idx, sdpa_q});
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
}

TEST_F(TransformationTestsF, IndirectKVCache5) {
    std::vector<int64_t> in0_order = {0, 1, 2, 3};
    std::vector<int64_t> in1_order = {0, 1, 2, 3};
    std::vector<int64_t> in2_order = {0, 1, 2, 3};
    std::vector<int64_t> out_order = {0, 1, 2, 3};
    const bool is_causal = false;
    {
        auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});
        auto key_variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{-1, 32, -1, 80}, ov::element::f16, "v0"});
        auto value_variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{-1, 32, -1, 80}, ov::element::f16, "v1"});
        auto key_past = std::make_shared<ov::intel_gpu::op::ReadValue>(key_variable);
        auto value_past = std::make_shared<ov::intel_gpu::op::ReadValue>(value_variable);
        auto axis = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, 0);
        auto key_gather_past = std::make_shared<ov::op::v8::Gather>(key_past, beam_idx, axis);
        auto value_gather_past = std::make_shared<ov::op::v8::Gather>(value_past, beam_idx, axis);

        auto key_data = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 32, -1, 240});
        auto vs_axis = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, 3);
        auto split_lengths = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{3}, std::vector<int64_t>{80, 80, -1});
        auto var_split = std::make_shared<ov::op::v1::VariadicSplit>(key_data, vs_axis, split_lengths);
        auto parameter_value = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 32, -1, 80});
        auto key_cache = std::make_shared<ov::intel_gpu::op::KVCache>(key_gather_past, var_split->output(0), key_variable, 0, ov::element::f16);
        auto value_cache = std::make_shared<ov::intel_gpu::op::KVCache>(value_gather_past, parameter_value, value_variable, 0, ov::element::f16);

        auto sdpa_q = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 32, -1, 80});
        auto attn_mask = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 1, -1, -1});
        auto scale = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{});
        auto inputs = ov::OutputVector{sdpa_q, key_cache, value_cache, attn_mask, scale};
        auto sdpa = std::make_shared<ov::intel_gpu::op::SDPA>(inputs, is_causal, in0_order, in1_order, in2_order, out_order);
        auto result = std::make_shared<ov::op::v0::Result>(sdpa);

        model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{key_data, parameter_value, beam_idx, sdpa_q, attn_mask, scale});
        manager.register_pass<IndirectKVCache>();
    }
    {
        auto indirect_axis = 0;
        auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});
        auto key_variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{-1, 32, -1, 80}, ov::element::f16, "v0"});
        auto value_variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{-1, 32, -1, 80}, ov::element::f16, "v1"});
        auto key_past = std::make_shared<ov::intel_gpu::op::ReadValue>(key_variable);
        auto value_past = std::make_shared<ov::intel_gpu::op::ReadValue>(value_variable);

        auto key_data = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 32, -1, 240});
        auto vs_axis = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, 3);
        auto split_lengths = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{3}, std::vector<int64_t>{80, 80, -1});
        auto var_split = std::make_shared<ov::op::v1::VariadicSplit>(key_data, vs_axis, split_lengths);
        auto parameter_value = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 32, -1, 80});
        auto key_cache = std::make_shared<ov::intel_gpu::op::KVCache>(key_past, var_split->output(0), beam_idx, key_variable, 0, 0, ov::element::f16);
        auto value_cache = std::make_shared<ov::intel_gpu::op::KVCache>(value_past, parameter_value, beam_idx, key_variable, 0, 0, ov::element::f16);

        auto sdpa_q = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 32, -1, 80});
        auto attn_mask = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 1, -1, -1});
        auto scale = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{});
        auto inputs = ov::OutputVector{sdpa_q, key_cache, value_cache, attn_mask, scale};

        auto sdpa = std::make_shared<ov::intel_gpu::op::IndirectSDPA>(inputs, key_cache->output(1), is_causal, indirect_axis, in0_order, in1_order, in2_order, out_order);
        auto result = std::make_shared<ov::op::v0::Result>(sdpa);

        model_ref = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{key_data, parameter_value, beam_idx, sdpa_q, attn_mask, scale});
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
}

TEST_F(TransformationTestsF, IndirectKVCacheSDPAWithSink) {
    const std::vector<int64_t> order = {0, 1, 2, 3};
    const bool is_causal = false;
    const auto make = [&](bool fused) {
        auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});
        auto key_variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{-1, 8, -1, 64}, ov::element::f16, "k"});
        auto value_variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{-1, 8, -1, 64}, ov::element::f16, "v"});
        auto key_new = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 8, -1, 64});
        auto value_new = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 8, -1, 64});
        auto query = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 8, -1, 64});
        auto mask = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 8, -1, -1});
        auto scale = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{});
        auto sink = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{1, 8, 1, 1});
        auto key_past = std::make_shared<ov::intel_gpu::op::ReadValue>(key_variable);
        auto value_past = std::make_shared<ov::intel_gpu::op::ReadValue>(value_variable);
        std::shared_ptr<ov::Node> sdpa;
        if (!fused) {
            auto axis = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, 0);
            auto key_cache = std::make_shared<ov::intel_gpu::op::KVCache>(std::make_shared<ov::op::v8::Gather>(key_past, beam_idx, axis), key_new, key_variable, 2, ov::element::f16);
            auto value_cache = std::make_shared<ov::intel_gpu::op::KVCache>(std::make_shared<ov::op::v8::Gather>(value_past, beam_idx, axis), value_new, value_variable, 2, ov::element::f16);
            sdpa = std::make_shared<ov::intel_gpu::op::SDPA>(ov::OutputVector{query, key_cache, value_cache, mask, scale, sink}, is_causal, order, order, order, order);
        } else {
            auto key_cache = std::make_shared<ov::intel_gpu::op::KVCache>(key_past, key_new, beam_idx, key_variable, 2, 0, ov::element::f16);
            auto value_cache = std::make_shared<ov::intel_gpu::op::KVCache>(value_past, value_new, beam_idx, value_variable, 2, 0, ov::element::f16);
            sdpa = std::make_shared<ov::intel_gpu::op::IndirectSDPA>(ov::OutputVector{query, key_cache, value_cache, mask, scale, sink}, key_cache->output(1), is_causal, 0, order, order, order, order);
        }
        return std::make_shared<ov::Model>(ov::OutputVector{sdpa}, ov::ParameterVector{beam_idx, key_new, value_new, query, mask, scale, sink});
    };
    model = make(false);
    manager.register_pass<IndirectKVCache>();
    model_ref = make(true);
    comparator.enable(FunctionsComparator::ATTRIBUTES);
}

namespace {

// ReadValue -> Gather(beam_idx) -> KVCache for the key and value of a SparseAttention, as KVCacheFusion leaves them.
struct SparseAttentionCaches {
    std::shared_ptr<ov::op::v0::Parameter> query, key, value, beam_idx, indices, count, key_length;
    std::shared_ptr<ov::op::util::Variable> key_variable, value_variable;
};

SparseAttentionCaches make_sparse_attention_caches() {
    SparseAttentionCaches c;
    const auto f16 = ov::element::f16;
    c.query = std::make_shared<ov::op::v0::Parameter>(f16, ov::PartialShape{-1, 4, -1, 32});
    c.key = std::make_shared<ov::op::v0::Parameter>(f16, ov::PartialShape{-1, 2, -1, 32});
    c.value = std::make_shared<ov::op::v0::Parameter>(f16, ov::PartialShape{-1, 2, -1, 32});
    c.beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});
    c.indices = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1, 1, -1, 8});
    c.count = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1, 1, -1});
    c.key_length = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});
    c.key_variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{-1, 2, -1, 32}, f16, "k"});
    c.value_variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{-1, 2, -1, 32}, f16, "v"});
    return c;
}

ov::ParameterVector parameters(const SparseAttentionCaches& c) {
    return {c.query, c.key, c.value, c.beam_idx, c.indices, c.count, c.key_length};
}

std::shared_ptr<ov::intel_gpu::op::KVCache> gathered_kv_cache(const SparseAttentionCaches& c,
                                                              const std::shared_ptr<ov::op::v0::Parameter>& new_token,
                                                              const std::shared_ptr<ov::op::util::Variable>& variable) {
    auto past = std::make_shared<ov::intel_gpu::op::ReadValue>(variable);
    auto gather = std::make_shared<ov::op::v8::Gather>(past, c.beam_idx, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0}));
    return std::make_shared<ov::intel_gpu::op::KVCache>(gather, new_token, variable, 2, ov::element::f16);
}

// The GPU SparseSDPA op ConvertSparseAttentionToInternal makes of a SparseAttention.
std::shared_ptr<ov::intel_gpu::op::SparseSDPA> sparse_attention(const SparseAttentionCaches& c,
                                                                const ov::Output<ov::Node>& key,
                                                                const ov::Output<ov::Node>& value) {
    ov::intel_gpu::op::SparseSDPA::Config config;
    config.num_heads = 4;
    config.num_kv_heads = 2;
    config.k_head_size = 32;
    config.v_head_size = 32;
    config.key_block_size = 4;
    return std::make_shared<ov::intel_gpu::op::SparseSDPA>(ov::OutputVector{c.query, key, value, c.indices, c.count, c.key_length}, config);
}

}  // namespace

// The SparseAttention caches become indirect, as the SDPA ones; the sparse kernels do not read the beam table, so the
// caches are marked for the identity beam_idx check of the infer request.
TEST_F(TransformationTestsF, IndirectKVCacheSparseAttention) {
    {
        const auto c = make_sparse_attention_caches();
        const auto attention = sparse_attention(c, gathered_kv_cache(c, c.key, c.key_variable), gathered_kv_cache(c, c.value, c.value_variable));
        model = std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(attention)}, parameters(c));
        manager.register_pass<IndirectKVCache>();
    }
    {
        const auto c = make_sparse_attention_caches();
        const auto indirect = [&c](const std::shared_ptr<ov::op::v0::Parameter>& new_token, const std::shared_ptr<ov::op::util::Variable>& variable) {
            auto past = std::make_shared<ov::intel_gpu::op::ReadValue>(variable);
            auto kv_cache = std::make_shared<ov::intel_gpu::op::KVCache>(past, new_token, c.beam_idx, variable, 2, 0, ov::element::f16);
            kv_cache->get_rt_info()[ov::intel_gpu::op::KV_CACHE_REQUIRES_IDENTITY_BEAM_IDX] = true;
            return kv_cache;
        };
        const auto attention = sparse_attention(c, indirect(c.key, c.key_variable)->output(0), indirect(c.value, c.value_variable)->output(0));
        model_ref = std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(attention)}, parameters(c));
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
}

// A cache also read by another node keeps its beam_idx Gather: that reader expects the reordered rows.
TEST_F(TransformationTestsF, IndirectKVCacheSparseAttentionSharedCache) {
    const auto c = make_sparse_attention_caches();
    const auto key_cache = gathered_kv_cache(c, c.key, c.key_variable);
    const auto attention = sparse_attention(c, key_cache, gathered_kv_cache(c, c.value, c.value_variable));
    model = std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(attention), std::make_shared<ov::op::v0::Result>(key_cache)},
                                        parameters(c));
    manager.register_pass<IndirectKVCache>();
}

// A shape read of a cache (the key_length ConvertSparseAttentionToInternal makes for a SparseAttention without one) does
// not depend on the row order: the cache still becomes indirect, and the shape read follows it.
TEST(IndirectKVCacheSparseAttentionTest, ShapeReadOfCacheKeepsItIndirect) {
    const auto c = make_sparse_attention_caches();
    const auto key_cache = gathered_kv_cache(c, c.key, c.key_variable);
    const auto key_shape = std::make_shared<ov::op::v3::ShapeOf>(key_cache, ov::element::i32);
    const auto attention = sparse_attention(c, key_cache, gathered_kv_cache(c, c.value, c.value_variable));
    const auto model =
        std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(attention), std::make_shared<ov::op::v0::Result>(key_shape)},
                                    parameters(c));
    ov::pass::Manager manager;
    manager.register_pass<IndirectKVCache>();
    manager.run_passes(model);
    const auto kv_cache = ov::as_type_ptr<ov::intel_gpu::op::KVCache>(attention->get_input_node_shared_ptr(ov::intel_gpu::op::SparseSDPA::InputIdx::KEY));
    ASSERT_NE(kv_cache, nullptr);
    EXPECT_TRUE(kv_cache->get_indirect());
    EXPECT_EQ(key_shape->get_input_node_shared_ptr(0), kv_cache);
}
