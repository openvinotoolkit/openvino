// Copyright (C) 2024 Intel Corporation
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
#include "openvino/pass/manager.hpp"

#include <transformations/utils/utils.hpp>
#include "plugin/transformations/indirect_kv_cache.hpp"

#include "intel_gpu/op/indirect_gemm.hpp"
#include "intel_gpu/op/gemm.hpp"
#include "intel_gpu/op/read_value.hpp"
#include "intel_gpu/op/kv_cache.hpp"

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
        auto gemm = std::make_shared<ov::intel_gpu::op::IndirectGemm>(gemm_in, kv_cache->output(0), kv_cache->output(1), false, true,
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
        auto gemm = std::make_shared<ov::intel_gpu::op::IndirectGemm>(kv_cache->output(0), gemm_in, kv_cache->output(1), true, false,
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
        auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1});
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
        model_ref = model->clone();
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
}
