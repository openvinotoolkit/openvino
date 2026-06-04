// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_test_utils.hpp"

#include <openvino/runtime/core.hpp>
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset13.hpp"
#include "openvino/pass/manager.hpp"

#include "plugin/transformations/transpose_fusion.hpp"

#include "ov_ops/vl_sdpa.hpp"

#include <openvino/pass/serialize.hpp>

#include <memory>

using namespace testing;
using namespace ov::intel_gpu;
using namespace ov;
using namespace ov::opset13;

namespace ov {
namespace test {
namespace intel_gpu {

namespace {

// Build the original model with Transpose + Split pattern from Qwen-VL Vision Merger
// This pattern appears in Qwen-VL models (Qwen2-VL, Qwen2.5-VL, Qwen3-VL, etc.)
// Pattern: Parameter[-1, 3, H, S] -> Transpose[3, -1, H, S] -> Split(axis=0) -> Reshape
std::shared_ptr<ov::Model> build_model_with_transpose_split() {
    const int64_t num_head = 8;
    const int64_t head_size = 32;

    // Parameter with shape [-1, 3, num_head, head_size]
    auto qkv = std::make_shared<Parameter>(element::f16, PartialShape{-1, 3, num_head, head_size});
    qkv->set_friendly_name("qkv");
    qkv->get_output_tensor(0).set_names({"qkv"});

    // Transpose: [-1, 3, H, S] -> [3, -1, H, S]
    auto transpose_order = Constant::create(element::i64, Shape{4}, std::vector<int64_t>{1, 0, 2, 3});
    auto transpose = std::make_shared<Transpose>(qkv, transpose_order);
    transpose->set_friendly_name("transpose_qkv");

    // Split along axis 0: [3, -1, H, S] -> 3x [1, -1, H, S]
    auto split_axis = Constant::create(element::i64, Shape{}, std::vector<int64_t>{0});
    auto split = std::make_shared<Split>(transpose, split_axis, 3);
    split->set_friendly_name("split_qkv");

    // Reshape each split output: [1, -1, H, S] -> [-1, H, S]
    auto reshape_pattern = Constant::create(element::i64, Shape{3}, std::vector<int64_t>{-1, num_head, head_size});

    auto reshape_q = std::make_shared<Reshape>(split->output(0), reshape_pattern, false);
    reshape_q->set_friendly_name("reshape_q");

    auto reshape_k = std::make_shared<Reshape>(split->output(1), reshape_pattern, false);
    reshape_k->set_friendly_name("reshape_k");

    auto reshape_v = std::make_shared<Reshape>(split->output(2), reshape_pattern, false);
    reshape_v->set_friendly_name("reshape_v");

    // RoPE for q and k (simplified using Multiply for testing)
    auto rope_cos_sin = Constant::create(element::f16, Shape{1, num_head, head_size}, {1.0f});

    auto rope_q = std::make_shared<Multiply>(reshape_q, rope_cos_sin);
    rope_q->set_friendly_name("rope_q");

    auto rope_k = std::make_shared<Multiply>(reshape_k, rope_cos_sin);
    rope_k->set_friendly_name("rope_k");

    // Create VLSDPA
    auto cu_seq_lens = std::make_shared<Parameter>(element::i32, PartialShape{-1});
    cu_seq_lens->set_friendly_name("cu_seq_lens");
    cu_seq_lens->get_output_tensor(0).set_names({"cu_seq_lens"});

    auto vlsdpa = std::make_shared<ov::op::internal::VLSDPA>(rope_q, rope_k, reshape_v, cu_seq_lens);
    vlsdpa->set_friendly_name("vlsdpa");

    return std::make_shared<ov::Model>(OutputVector{vlsdpa}, ParameterVector{qkv, cu_seq_lens});
}

// Build the target model after TransposeSplitMatcher transformation
// Pattern: Parameter[-1, 3, H, S] -> Split(axis=1) -> Reshape
std::shared_ptr<ov::Model> build_target_model_with_optimized_split() {
    const int64_t num_head = 8;
    const int64_t head_size = 32;

    // Parameter with shape [-1, 3, num_head, head_size]
    auto qkv = std::make_shared<Parameter>(element::f16, PartialShape{-1, 3, num_head, head_size});
    qkv->set_friendly_name("qkv");
    qkv->get_output_tensor(0).set_names({"qkv"});

    // Split along axis 1: [-1, 3, H, S] -> 3x [-1, 1, H, S]
    auto split_axis = Constant::create(element::i64, Shape{}, std::vector<int64_t>{1});
    auto split = std::make_shared<Split>(qkv, split_axis, 3);
    split->set_friendly_name("split_qkv");

    // Reshape each split output: [-1, 1, H, S] -> [-1, H, S]
    auto reshape_pattern = Constant::create(element::i64, Shape{3}, std::vector<int64_t>{-1, num_head, head_size});

    auto reshape_q = std::make_shared<Reshape>(split->output(0), reshape_pattern, false);
    reshape_q->set_friendly_name("reshape_q");

    auto reshape_k = std::make_shared<Reshape>(split->output(1), reshape_pattern, false);
    reshape_k->set_friendly_name("reshape_k");

    auto reshape_v = std::make_shared<Reshape>(split->output(2), reshape_pattern, false);
    reshape_v->set_friendly_name("reshape_v");

    // RoPE for q and k (simplified using Multiply for testing)
    auto rope_cos_sin = Constant::create(element::f16, Shape{1, num_head, head_size}, {1.0f});

    auto rope_q = std::make_shared<Multiply>(reshape_q, rope_cos_sin);
    rope_q->set_friendly_name("rope_q");

    auto rope_k = std::make_shared<Multiply>(reshape_k, rope_cos_sin);
    rope_k->set_friendly_name("rope_k");

    // Create VLSDPA
    auto cu_seq_lens = std::make_shared<Parameter>(element::i32, PartialShape{-1});
    cu_seq_lens->set_friendly_name("cu_seq_lens");
    cu_seq_lens->get_output_tensor(0).set_names({"cu_seq_lens"});

    auto vlsdpa = std::make_shared<ov::op::internal::VLSDPA>(rope_q, rope_k, reshape_v, cu_seq_lens);
    vlsdpa->set_friendly_name("vlsdpa");

    return std::make_shared<ov::Model>(OutputVector{vlsdpa}, ParameterVector{qkv, cu_seq_lens});
}

}  // namespace

TEST_F(TransformationTestsF, TransposeSplitFusionTest) {
    disable_rt_info_check();
    {
        model = build_model_with_transpose_split();
        manager.register_pass<TransposeFusion>();
    }
    { model_ref = build_target_model_with_optimized_split(); }
}

}  // namespace intel_gpu
}  // namespace test
}  // namespace ov
