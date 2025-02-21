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

#include "intel_gpu/op/gemm.hpp"
#include "intel_gpu/op/read_value.hpp"
#include "intel_gpu/op/kv_cache.hpp"

#include "plugin/transformations/unsqueeze_broadcast_reshape_matmul_fusion.hpp"

#include <memory>

using namespace testing;
using namespace ov::intel_gpu;

namespace ov {
namespace test {
namespace intel_gpu {

TEST_F(TransformationTestsF, UnsqueezeBroadReshapeMatmulFusion1) {
    std::vector<int64_t> order_a = {0, 1, 2, 3};
    std::vector<int64_t> order_b = {1, 2, 3, 0};
    std::vector<int64_t> order_c = {0, 1, 2, 3};
    std::vector<int64_t> axes_b = {-2};
    std::vector<int32_t> target_shape_b = {1, 1, 1, 16, 1};
    std::vector<int64_t> pattern_b = {0, 0, 32, 32};
    {
        auto input_a = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{-1, -1, 2, 32}, ov::element::f32, "v0"});
        auto new_token_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, 2, 32});
        auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1});
        auto past = std::make_shared<ov::intel_gpu::op::ReadValue>(variable);
        auto axis = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, 0);
        auto gather_past = std::make_shared<ov::op::v8::Gather>(past, beam_idx, axis);
        auto kv_cache = std::make_shared<ov::intel_gpu::op::KVCache>(gather_past, new_token_param, variable, 2, ov::element::f32);
        auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, axes_b);
        auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(kv_cache, axes);
        auto target_shape = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{5}, target_shape_b);
        auto broadcast = std::make_shared<ov::op::v3::Broadcast>(unsqueeze, target_shape, ov::op::BroadcastType::BIDIRECTIONAL);
        auto pattern = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, pattern_b);
        auto reshape = std::make_shared<ov::op::v1::Reshape>(broadcast, pattern, true);
        auto gemm = std::make_shared<ov::intel_gpu::op::Gemm>(input_a,
                                                              reshape,
                                                              order_a,
                                                              order_b,
                                                              order_c,
                                                              ov::element::dynamic);

        model = std::make_shared<ov::Model>(ov::NodeVector{ gemm }, ov::ParameterVector{ input_a, new_token_param, beam_idx });
        manager.register_pass<UnsqueezeBroadcastReshapeMatmulFusion>();
    }
    {
        auto input_a = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{-1, -1, 2, 32}, ov::element::f32, "v0"});
        auto new_token_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, 2, 32});
        auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1});
        auto past = std::make_shared<ov::intel_gpu::op::ReadValue>(variable);
        auto axis = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, 0);
        auto gather_past = std::make_shared<ov::op::v8::Gather>(past, beam_idx, axis);
        auto kv_cache = std::make_shared<ov::intel_gpu::op::KVCache>(gather_past, new_token_param, variable, 2, ov::element::f32);
        auto gemm = std::make_shared<ov::intel_gpu::op::Gemm>(input_a,
                                                              kv_cache,
                                                              order_a,
                                                              order_b,
                                                              order_c,
                                                              ov::element::dynamic);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{ gemm }, ov::ParameterVector{ input_a, new_token_param, beam_idx });
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
}

TEST_F(TransformationTestsF, UnsqueezeBroadReshapeMatmulFusion2) {
    std::vector<int64_t> order_a = {0, 1, 2, 3};
    std::vector<int64_t> order_b = {0, 1, 3, 2};
    std::vector<int64_t> order_c = {0, 1, 2, 3};
    std::vector<int64_t> axes_b = {2};
    std::vector<int64_t> pattern_b = {0, 32, -1, 32};
    {
        auto input_a = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{-1, 8, -1, 32}, ov::element::f32, "v0"});
        auto new_token_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 8, -1, 32});
        auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1});
        auto past = std::make_shared<ov::intel_gpu::op::ReadValue>(variable);
        auto axis = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, 0);
        auto gather_past = std::make_shared<ov::op::v8::Gather>(past, beam_idx, axis);
        auto kv_cache = std::make_shared<ov::intel_gpu::op::KVCache>(gather_past, new_token_param, variable, 2, ov::element::f32);
        auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, axes_b);
        auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(kv_cache, axes);
        auto abs_param = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{5});
        auto abs = std::make_shared<ov::op::v0::Abs>(abs_param);
        auto broadcast = std::make_shared<ov::op::v3::Broadcast>(unsqueeze, abs, ov::op::BroadcastType::BIDIRECTIONAL);
        auto pattern = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, pattern_b);
        auto reshape = std::make_shared<ov::op::v1::Reshape>(broadcast, pattern, true);
        auto gemm = std::make_shared<ov::intel_gpu::op::Gemm>(input_a,
                                                              reshape,
                                                              order_a,
                                                              order_b,
                                                              order_c,
                                                              ov::element::dynamic);

        model = std::make_shared<ov::Model>(ov::NodeVector{ gemm }, ov::ParameterVector{ input_a, new_token_param, beam_idx, abs_param });
        manager.register_pass<UnsqueezeBroadcastReshapeMatmulFusion>();
    }
    {
        auto input_a = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{-1, 8, -1, 32}, ov::element::f32, "v0"});
        auto new_token_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 8, -1, 32});
        auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1});
        auto past = std::make_shared<ov::intel_gpu::op::ReadValue>(variable);
        auto axis = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, 0);
        auto gather_past = std::make_shared<ov::op::v8::Gather>(past, beam_idx, axis);
        auto kv_cache = std::make_shared<ov::intel_gpu::op::KVCache>(gather_past, new_token_param, variable, 2, ov::element::f32);
        auto gemm = std::make_shared<ov::intel_gpu::op::Gemm>(input_a,
                                                              kv_cache,
                                                              order_a,
                                                              order_b,
                                                              order_c,
                                                              ov::element::dynamic);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{ gemm }, ov::ParameterVector{ input_a, new_token_param, beam_idx });
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
}

TEST_F(TransformationTestsF, UnsqueezeBroadReshapeMatmulFusion3) {
    std::vector<int64_t> order_a = {0, 1, 2, 3};
    std::vector<int64_t> order_b = {1, 2, 3, 0};
    std::vector<int64_t> order_c = {0, 1, 2, 3};
    std::vector<int64_t> axes_b = {-2};
    std::vector<int32_t> target_shape_b = {1, 1, 1, 16, 1};
    std::vector<int64_t> pattern_b = {0, 0, -1, 32};
    {
        auto input_a = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto input_b = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, -1, 32});
        auto unsqueeze_b_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, axes_b);
        auto unsqueeze_b = std::make_shared<ov::op::v0::Unsqueeze>(input_b, unsqueeze_b_const);
        auto broadcast_b_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{5}, target_shape_b);
        auto broadcast_b = std::make_shared<ov::op::v3::Broadcast>(unsqueeze_b, broadcast_b_const, ov::op::BroadcastType::BIDIRECTIONAL);
        auto reshape_b_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, pattern_b);
        auto reshape_b = std::make_shared<ov::op::v1::Reshape>(broadcast_b, reshape_b_const, true);
        auto gemm = std::make_shared<ov::intel_gpu::op::Gemm>(input_a,
                                                              reshape_b,
                                                              order_a,
                                                              order_b,
                                                              order_c,
                                                              ov::element::dynamic);

        model = std::make_shared<ov::Model>(ov::NodeVector{ gemm }, ov::ParameterVector{ input_a, input_b });
        manager.register_pass<UnsqueezeBroadcastReshapeMatmulFusion>();
    }
    {
        model_ref = model->clone();
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
}

TEST_F(TransformationTestsF, UnsqueezeBroadReshapeMatmulFusion4) {
    std::vector<int64_t> order_a = {0, 1, 2, 3};
    std::vector<int64_t> order_b = {0, 1, 2, 3};
    std::vector<int64_t> order_c = {0, 1, 2, 3};
    std::vector<int64_t> axes_b = {-3};
    std::vector<int32_t> target_shape_b = {1, 1, 16, 1, 1};
    std::vector<int64_t> pattern_b = {0, 32, 32, 0};
    {
        auto input_a = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto input_b = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 2, 32, -1});
        auto unsqueeze_b_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, axes_b);
        auto unsqueeze_b = std::make_shared<ov::op::v0::Unsqueeze>(input_b, unsqueeze_b_const);
        auto broadcast_b_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{5}, target_shape_b);
        auto broadcast_b = std::make_shared<ov::op::v3::Broadcast>(unsqueeze_b, broadcast_b_const, ov::op::BroadcastType::BIDIRECTIONAL);
        auto reshape_b_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, pattern_b);
        auto reshape_b = std::make_shared<ov::op::v1::Reshape>(broadcast_b, reshape_b_const, true);
        auto gemm = std::make_shared<ov::intel_gpu::op::Gemm>(input_a,
                                                              reshape_b,
                                                              order_a,
                                                              order_b,
                                                              order_c,
                                                              ov::element::dynamic);

        model = std::make_shared<ov::Model>(ov::NodeVector{ gemm }, ov::ParameterVector{ input_a, input_b });
        manager.register_pass<UnsqueezeBroadcastReshapeMatmulFusion>();
    }
    {
        model_ref = model->clone();
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
}

}  // namespace intel_gpu
}  // namespace test
}  // namespace ov
