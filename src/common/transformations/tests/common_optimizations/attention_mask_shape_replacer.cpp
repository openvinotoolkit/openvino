// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/common_optimizations/broadcast_matmul_fusion.hpp"

namespace {

using namespace ov;
namespace v0 = ov::op::v0;
namespace v3 = ov::op::v3;
namespace v8 = ov::op::v8;

std::shared_ptr<v0::Parameter> make_attention_mask(const PartialShape& shape) {
    auto attention_mask = std::make_shared<v0::Parameter>(element::i64, shape);
    attention_mask->set_friendly_name("attention_mask");
    attention_mask->output(0).set_names({"attention_mask"});
    return attention_mask;
}

// Gemma-style rotary embedding batch broadcast:
// attention_mask -> ShapeOf -> Gather(batch, axis=0) -> Concat -> Broadcast -> MatMul(position_ids) -> Result
// In PagedAttention mode attention_mask is removed, so BroadcastMatMulFusion must be able to detach the
// Broadcast from the MatMul whenever the other operand provably carries the same batch dimension.
std::shared_ptr<ov::Model> build_model(const std::shared_ptr<v0::Parameter>& attention_mask,
                                       const std::shared_ptr<v0::Parameter>& position_ids,
                                       const Shape& broadcast_data_shape,
                                       bool with_broadcast) {
    auto broadcast_data = std::make_shared<v0::Parameter>(element::f32, broadcast_data_shape);
    broadcast_data->set_friendly_name("broadcast_data");

    Output<Node> matmul_lhs = broadcast_data;
    if (with_broadcast) {
        auto shape_of = std::make_shared<v3::ShapeOf>(attention_mask, element::i64);
        auto indices = v0::Constant::create(element::i64, Shape{1}, {0});
        auto axis = v0::Constant::create(element::i64, Shape{}, {0});
        auto gather = std::make_shared<v8::Gather>(shape_of, indices, axis);

        auto one = v0::Constant::create(element::i64, Shape{1}, {1});
        auto four = v0::Constant::create(element::i64, Shape{1}, {4});
        auto target_shape = std::make_shared<v0::Concat>(OutputVector{gather, one, four}, 0);
        matmul_lhs = std::make_shared<v3::Broadcast>(broadcast_data, target_shape, op::BroadcastType::BIDIRECTIONAL);
    }

    auto matmul = std::make_shared<v0::MatMul>(matmul_lhs, position_ids, false, false);
    auto result = std::make_shared<v0::Result>(matmul);
    return std::make_shared<ov::Model>(ResultVector{result},
                                       ParameterVector{attention_mask, position_ids, broadcast_data});
}

}  // namespace

class AttentionMaskBroadcastMatMulFusionTest : public ::TransformationTestsF {
protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        manager.register_pass<ov::pass::BroadcastMatMulFusion>();
    }
};

TEST_F(AttentionMaskBroadcastMatMulFusionTest, RemovesBroadcastForGemmaRotaryEmbeddingPattern) {
    // attention_mask and position_ids share the same static batch (2): the Broadcast can be
    // detached from the MatMul, so the rotary embedding no longer depends on attention_mask,
    // which is exactly what is required once attention_mask is removed in PagedAttention mode.
    auto attention_mask = make_attention_mask(PartialShape{2, -1});
    auto position_ids = std::make_shared<v0::Parameter>(element::f32, PartialShape{2, 4, 1});
    position_ids->set_friendly_name("position_ids");
    position_ids->output(0).set_names({"position_ids"});
    model = build_model(attention_mask, position_ids, Shape{1, 1, 4}, /*with_broadcast=*/true);

    auto attention_mask_ref = make_attention_mask(PartialShape{2, -1});
    auto position_ids_ref = std::make_shared<v0::Parameter>(element::f32, PartialShape{2, 4, 1});
    position_ids_ref->set_friendly_name("position_ids");
    position_ids_ref->output(0).set_names({"position_ids"});
    model_ref = build_model(attention_mask_ref, position_ids_ref, Shape{1, 1, 4}, /*with_broadcast=*/false);
}

TEST_F(AttentionMaskBroadcastMatMulFusionTest, KeepsBroadcastWhenPositionIdsBatchDoesNotCarryAttentionMaskBatch) {
    // position_ids' batch (1) cannot reproduce the Broadcast's expanded batch (derived from
    // attention_mask, dynamic here): the Broadcast must stay, since attention_mask still exists
    // in this (pre-PagedAttention) model.
    auto attention_mask = make_attention_mask(PartialShape{-1, -1});
    auto position_ids = std::make_shared<v0::Parameter>(element::f32, PartialShape{1, 4, 1});
    position_ids->set_friendly_name("position_ids");
    position_ids->output(0).set_names({"position_ids"});
    model = build_model(attention_mask, position_ids, Shape{1, 1, 4}, /*with_broadcast=*/true);
}
