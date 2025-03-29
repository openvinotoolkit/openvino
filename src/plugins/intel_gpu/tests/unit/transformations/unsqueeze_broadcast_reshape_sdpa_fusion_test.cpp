// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_test_utils.hpp"

#include "openvino/core/model.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/unsqueeze.hpp"

#include "intel_gpu/op/sdpa.hpp"
#include "intel_gpu/op/read_value.hpp"
#include "intel_gpu/op/kv_cache.hpp"

#include "plugin/transformations/unsqueeze_broadcast_reshape_sdpa_fusion.hpp"

#include <memory>

using namespace testing;
using namespace ov::intel_gpu;

namespace ov {
namespace test {
namespace intel_gpu {

TEST_F(TransformationTestsF, UnsqueezeBroadReshapeSDPAFusion1) {
    std::vector<int64_t> in0_order = {0, 2, 1, 3};
    std::vector<int64_t> in1_order = {0, 2, 1, 3};
    std::vector<int64_t> in2_order = {0, 2, 1, 3};
    std::vector<int64_t> out_order = {0, 1, 2, 3};
    std::vector<int64_t> axes_val = {-2};
    std::vector<int32_t> target_shape_kv = {1, 1, 1, 16, 1};
    std::vector<int64_t> pattern_shape = {0, 0, 32, 32};
    const bool is_causal = true;
    {
        auto input_q = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto key_variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{-1, -1, 2, 32}, ov::element::f32, "v0"});
        auto value_variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{-1, -1, 2, 32}, ov::element::f32, "v1"});
        auto key_token_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, 2, 32});
        auto value_token_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, 2, 32});
        auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1});
        auto key_past = std::make_shared<ov::intel_gpu::op::ReadValue>(key_variable);
        auto value_past = std::make_shared<ov::intel_gpu::op::ReadValue>(key_variable);
        auto axis = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, 0);
        auto key_gather_past = std::make_shared<ov::op::v8::Gather>(key_past, beam_idx, axis);
        auto value_gather_past = std::make_shared<ov::op::v8::Gather>(value_past, beam_idx, axis);
        auto key_cache = std::make_shared<ov::intel_gpu::op::KVCache>(key_gather_past, key_token_param, key_variable, 2, ov::element::f32);
        auto value_cache = std::make_shared<ov::intel_gpu::op::KVCache>(value_gather_past, value_token_param, value_variable, 2, ov::element::f32);
        auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, axes_val);
        auto key_unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(key_cache, axes);
        auto value_unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(value_cache, axes);
        auto target_shape = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{5}, target_shape_kv);
        auto key_broadcast = std::make_shared<ov::op::v3::Broadcast>(key_unsqueeze, target_shape, ov::op::BroadcastType::BIDIRECTIONAL);
        auto value_broadcast = std::make_shared<ov::op::v3::Broadcast>(value_unsqueeze, target_shape, ov::op::BroadcastType::BIDIRECTIONAL);
        auto pattern = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, pattern_shape);
        auto key_reshape = std::make_shared<ov::op::v1::Reshape>(key_broadcast, pattern, true);
        auto value_reshape = std::make_shared<ov::op::v1::Reshape>(value_broadcast, pattern, true);
        auto inputs = ov::OutputVector{input_q, key_reshape, value_reshape};
        auto sdpa = std::make_shared<ov::intel_gpu::op::SDPA>(inputs, is_causal, in0_order, in1_order, in2_order, out_order);

        model = std::make_shared<ov::Model>(ov::NodeVector{ sdpa }, ov::ParameterVector{ input_q, key_token_param, value_token_param, beam_idx });
        manager.register_pass<UnsqueezeBroadcastReshapeSDPAFusion>();
    }
    {
        auto input_q = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto key_variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{-1, -1, 2, 32}, ov::element::f32, "v0"});
        auto value_variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{-1, -1, 2, 32}, ov::element::f32, "v1"});
        auto key_token_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, 2, 32});
        auto value_token_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, 2, 32});
        auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1});
        auto key_past = std::make_shared<ov::intel_gpu::op::ReadValue>(key_variable);
        auto value_past = std::make_shared<ov::intel_gpu::op::ReadValue>(key_variable);
        auto axis = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, 0);
        auto key_gather_past = std::make_shared<ov::op::v8::Gather>(key_past, beam_idx, axis);
        auto value_gather_past = std::make_shared<ov::op::v8::Gather>(value_past, beam_idx, axis);
        auto key_cache = std::make_shared<ov::intel_gpu::op::KVCache>(key_gather_past, key_token_param, key_variable, 2, ov::element::f32);
        auto value_cache = std::make_shared<ov::intel_gpu::op::KVCache>(value_gather_past, value_token_param, value_variable, 2, ov::element::f32);
        auto inputs = ov::OutputVector{input_q, key_cache, value_cache};
        auto sdpa = std::make_shared<ov::intel_gpu::op::SDPA>(inputs, is_causal, in0_order, in1_order, in2_order, out_order);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{ sdpa }, ov::ParameterVector{ input_q, key_token_param, value_token_param, beam_idx });
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
}

TEST_F(TransformationTestsF, UnsqueezeBroadReshapeSDPAFusion2) {
    std::vector<int64_t> in0_order = {0, 2, 1, 3};
    std::vector<int64_t> in1_order = {0, 2, 1, 3};
    std::vector<int64_t> in2_order = {0, 2, 1, 3};
    std::vector<int64_t> out_order = {0, 1, 2, 3};
    std::vector<int64_t> axes_val = {2};
    std::vector<int64_t> pattern_shape = {0, 32, -1, 32};
    const bool is_causal = true;
    {
        auto input_q = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto key_variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{-1, 8, -1, 32}, ov::element::f32, "v0"});
        auto value_variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{-1, 8, -1, 32}, ov::element::f32, "v1"});
        auto key_token_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 8, -1, 32});
        auto value_token_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 8, -1, 32});
        auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1});
        auto key_past = std::make_shared<ov::intel_gpu::op::ReadValue>(key_variable);
        auto value_past = std::make_shared<ov::intel_gpu::op::ReadValue>(key_variable);
        auto axis = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, 0);
        auto key_gather_past = std::make_shared<ov::op::v8::Gather>(key_past, beam_idx, axis);
        auto value_gather_past = std::make_shared<ov::op::v8::Gather>(value_past, beam_idx, axis);
        auto key_cache = std::make_shared<ov::intel_gpu::op::KVCache>(key_gather_past, key_token_param, key_variable, 2, ov::element::f32);
        auto value_cache = std::make_shared<ov::intel_gpu::op::KVCache>(value_gather_past, value_token_param, value_variable, 2, ov::element::f32);
        auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, axes_val);
        auto key_unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(key_cache, axes);
        auto value_unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(value_cache, axes);
        auto abs_param = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{5});
        auto abs = std::make_shared<ov::op::v0::Abs>(abs_param);
        auto key_broadcast = std::make_shared<ov::op::v3::Broadcast>(key_unsqueeze, abs, ov::op::BroadcastType::BIDIRECTIONAL);
        auto value_broadcast = std::make_shared<ov::op::v3::Broadcast>(value_unsqueeze, abs, ov::op::BroadcastType::BIDIRECTIONAL);
        auto pattern = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, pattern_shape);
        auto key_reshape = std::make_shared<ov::op::v1::Reshape>(key_broadcast, pattern, true);
        auto value_reshape = std::make_shared<ov::op::v1::Reshape>(value_broadcast, pattern, true);
        auto inputs = ov::OutputVector{input_q, key_cache, value_cache};
        auto sdpa = std::make_shared<ov::intel_gpu::op::SDPA>(inputs, is_causal, in0_order, in1_order, in2_order, out_order);

        model = std::make_shared<ov::Model>(ov::NodeVector{ sdpa }, ov::ParameterVector{ input_q, key_token_param, value_token_param, beam_idx });
        manager.register_pass<UnsqueezeBroadcastReshapeSDPAFusion>();
    }
    {
        auto input_q = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto key_variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{-1, 8, -1, 32}, ov::element::f32, "v0"});
        auto value_variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{-1, 8, -1, 32}, ov::element::f32, "v1"});
        auto key_token_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 8, -1, 32});
        auto value_token_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 8, -1, 32});
        auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1});
        auto key_past = std::make_shared<ov::intel_gpu::op::ReadValue>(key_variable);
        auto value_past = std::make_shared<ov::intel_gpu::op::ReadValue>(key_variable);
        auto axis = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, 0);
        auto key_gather_past = std::make_shared<ov::op::v8::Gather>(key_past, beam_idx, axis);
        auto value_gather_past = std::make_shared<ov::op::v8::Gather>(value_past, beam_idx, axis);
        auto key_cache = std::make_shared<ov::intel_gpu::op::KVCache>(key_gather_past, key_token_param, key_variable, 2, ov::element::f32);
        auto value_cache = std::make_shared<ov::intel_gpu::op::KVCache>(value_gather_past, value_token_param, value_variable, 2, ov::element::f32);
        auto inputs = ov::OutputVector{input_q, key_cache, value_cache};
        auto sdpa = std::make_shared<ov::intel_gpu::op::SDPA>(inputs, is_causal, in0_order, in1_order, in2_order, out_order);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{ sdpa }, ov::ParameterVector{ input_q, key_token_param, value_token_param, beam_idx });
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
}

TEST_F(TransformationTestsF, UnsqueezeBroadReshapeSDPAFusion3) {
    std::vector<int64_t> in0_order = {0, 2, 1, 3};
    std::vector<int64_t> in1_order = {0, 2, 1, 3};
    std::vector<int64_t> in2_order = {0, 2, 1, 3};
    std::vector<int64_t> out_order = {0, 1, 2, 3};
    std::vector<int64_t> axes_val = {-2};
    std::vector<int32_t> target_shape_kv = {1, 1, 1, 16, 1};
    std::vector<int64_t> pattern_shape = {0, 0, -1, 32};
    const bool is_causal = true;
    {
        auto input_q = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto input_k = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, -1, 32});
        auto input_v = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, -1, 32});
        auto unsqueeze_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, axes_val);
        auto key_unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(input_k, unsqueeze_const);
        auto value_unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(input_v, unsqueeze_const);
        auto broadcast_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{5}, target_shape_kv);
        auto key_broadcast = std::make_shared<ov::op::v3::Broadcast>(key_unsqueeze, broadcast_const, ov::op::BroadcastType::BIDIRECTIONAL);
        auto value_broadcast = std::make_shared<ov::op::v3::Broadcast>(value_unsqueeze, broadcast_const, ov::op::BroadcastType::BIDIRECTIONAL);
        auto reshape_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, pattern_shape);
        auto key_reshape = std::make_shared<ov::op::v1::Reshape>(key_broadcast, reshape_const, true);
        auto value_reshape = std::make_shared<ov::op::v1::Reshape>(value_broadcast, reshape_const, true);
        auto inputs = ov::OutputVector{input_q, key_reshape, value_reshape};
        auto sdpa = std::make_shared<ov::intel_gpu::op::SDPA>(inputs, is_causal, in0_order, in1_order, in2_order, out_order);

        model = std::make_shared<ov::Model>(ov::NodeVector{ sdpa }, ov::ParameterVector{ input_q, input_k, input_v });
        manager.register_pass<UnsqueezeBroadcastReshapeSDPAFusion>();
    }
    {
        model_ref = model->clone();
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
}

TEST_F(TransformationTestsF, UnsqueezeBroadReshapeSDPAFusion4) {
    std::vector<int64_t> in0_order = {0, 2, 1, 3};
    std::vector<int64_t> in1_order = {0, 2, 1, 3};
    std::vector<int64_t> in2_order = {0, 2, 1, 3};
    std::vector<int64_t> out_order = {0, 1, 2, 3};
    std::vector<int64_t> axes_val = {-2};
    std::vector<int32_t> target_shape_k = {1, 1, 1, 16, 1};
    std::vector<int32_t> target_shape_v = {1, 1, 1, 32, 1};
    std::vector<int64_t> pattern_shape = {0, 0, 32, 32};
    const bool is_causal = true;
    {
        auto input_q = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto key_variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{-1, -1, 2, 32}, ov::element::f32, "v0"});
        auto value_variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{-1, -1, 2, 32}, ov::element::f32, "v1"});
        auto key_token_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, 2, 32});
        auto value_token_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, 2, 32});
        auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1});
        auto key_past = std::make_shared<ov::intel_gpu::op::ReadValue>(key_variable);
        auto value_past = std::make_shared<ov::intel_gpu::op::ReadValue>(key_variable);
        auto axis = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, 0);
        auto key_gather_past = std::make_shared<ov::op::v8::Gather>(key_past, beam_idx, axis);
        auto value_gather_past = std::make_shared<ov::op::v8::Gather>(value_past, beam_idx, axis);
        auto key_cache = std::make_shared<ov::intel_gpu::op::KVCache>(key_gather_past, key_token_param, key_variable, 2, ov::element::f32);
        auto value_cache = std::make_shared<ov::intel_gpu::op::KVCache>(value_gather_past, value_token_param, value_variable, 2, ov::element::f32);
        auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, axes_val);
        auto key_unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(key_cache, axes);
        auto value_unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(value_cache, axes);
        auto target_shape_key = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{5}, target_shape_k);
        auto target_shape_value = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{5}, target_shape_v);
        auto key_broadcast = std::make_shared<ov::op::v3::Broadcast>(key_unsqueeze, target_shape_key, ov::op::BroadcastType::BIDIRECTIONAL);
        auto value_broadcast = std::make_shared<ov::op::v3::Broadcast>(value_unsqueeze, target_shape_value, ov::op::BroadcastType::BIDIRECTIONAL);
        auto pattern = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, pattern_shape);
        auto key_reshape = std::make_shared<ov::op::v1::Reshape>(key_broadcast, pattern, true);
        auto value_reshape = std::make_shared<ov::op::v1::Reshape>(value_broadcast, pattern, true);
        auto inputs = ov::OutputVector{input_q, key_reshape, value_reshape};
        auto sdpa = std::make_shared<ov::intel_gpu::op::SDPA>(inputs, is_causal, in0_order, in1_order, in2_order, out_order);

        model = std::make_shared<ov::Model>(ov::NodeVector{ sdpa }, ov::ParameterVector{ input_q, key_token_param, value_token_param, beam_idx });
        manager.register_pass<UnsqueezeBroadcastReshapeSDPAFusion>();
    }
    {
        model_ref = model->clone();
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
}

}  // namespace intel_gpu
}  // namespace test
}  // namespace ov
