// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/paged_attention/attention_mask_shape_replacer.hpp"

#include <gtest/gtest.h>

#include <vector>

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

namespace {

using namespace ov;
namespace v0 = ov::op::v0;
namespace v3 = ov::op::v3;
namespace v8 = ov::op::v8;

std::shared_ptr<v0::Parameter> make_attention_mask() {
    auto attention_mask = std::make_shared<v0::Parameter>(element::i64, PartialShape{-1, -1});
    attention_mask->set_friendly_name("attention_mask");
    attention_mask->output(0).set_names({"attention_mask"});
    return attention_mask;
}

// attention_mask -> ShapeOf -> Gather(indices, axis=0) -> Concat -> Broadcast -> MatMul(position_ids) -> Result
std::shared_ptr<ov::Model> build_model(const Output<Node>& shape_source,
                                       const std::shared_ptr<v0::Parameter>& attention_mask,
                                       const std::shared_ptr<v0::Parameter>& input_source,
                                       const std::shared_ptr<v0::Parameter>& position_ids,
                                       const std::vector<int64_t>& gather_indices) {
    auto shape_of = std::make_shared<v3::ShapeOf>(shape_source, element::i64);
    auto indices = v0::Constant::create(element::i64, Shape{gather_indices.size()}, gather_indices);
    auto axis = v0::Constant::create(element::i64, Shape{}, {0});
    auto gather = std::make_shared<v8::Gather>(shape_of, indices, axis);

    auto one = v0::Constant::create(element::i64, Shape{1}, {1});
    auto four = v0::Constant::create(element::i64, Shape{1}, {4});
    auto target_shape = std::make_shared<v0::Concat>(OutputVector{gather, one, four}, 0);

    auto broadcast_data = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, 1, 4});
    broadcast_data->set_friendly_name("broadcast_data");
    auto broadcast = std::make_shared<v3::Broadcast>(broadcast_data, target_shape);

    auto matmul = std::make_shared<v0::MatMul>(broadcast, position_ids, false, false);
    auto result = std::make_shared<v0::Result>(matmul);
    return std::make_shared<ov::Model>(ResultVector{result},
                                       ParameterVector{attention_mask, input_source, position_ids, broadcast_data});
}

}  // namespace

class AttentionMaskShapeReplacerTest : public ::TransformationTestsF {};

TEST_F(AttentionMaskShapeReplacerTest, ReplacesWithInputIdsBatchDim) {
    auto attention_mask = make_attention_mask();
    auto input_ids = std::make_shared<v0::Parameter>(element::i32, PartialShape{-1, -1});
    input_ids->set_friendly_name("input_ids");
    auto position_ids = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, 4, 1});
    position_ids->set_friendly_name("position_ids");
    position_ids->output(0).set_names({"position_ids"});
    model = build_model(attention_mask, attention_mask, input_ids, position_ids, {0});

    manager.register_pass<ov::pass::AttentionMaskShapeReplacer>(input_ids);

    auto attention_mask_ref = make_attention_mask();
    auto input_ids_ref = std::make_shared<v0::Parameter>(element::i32, PartialShape{-1, -1});
    input_ids_ref->set_friendly_name("input_ids");
    auto position_ids_ref = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, 4, 1});
    position_ids_ref->set_friendly_name("position_ids");
    position_ids_ref->output(0).set_names({"position_ids"});
    model_ref = build_model(input_ids_ref, attention_mask_ref, input_ids_ref, position_ids_ref, {0});
}

TEST_F(AttentionMaskShapeReplacerTest, ReplacesWithInputsEmbedsBatchDim) {
    auto attention_mask = make_attention_mask();
    auto inputs_embeds = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, -1, 2048});
    inputs_embeds->set_friendly_name("inputs_embeds");
    auto position_ids = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, 4, 1});
    position_ids->set_friendly_name("position_ids");
    position_ids->output(0).set_names({"position_ids"});
    model = build_model(attention_mask, attention_mask, inputs_embeds, position_ids, {0});

    manager.register_pass<ov::pass::AttentionMaskShapeReplacer>(inputs_embeds);

    auto attention_mask_ref = make_attention_mask();
    auto inputs_embeds_ref = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, -1, 2048});
    inputs_embeds_ref->set_friendly_name("inputs_embeds");
    auto position_ids_ref = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, 4, 1});
    position_ids_ref->set_friendly_name("position_ids");
    position_ids_ref->output(0).set_names({"position_ids"});
    model_ref = build_model(inputs_embeds_ref, attention_mask_ref, inputs_embeds_ref, position_ids_ref, {0});
}

TEST_F(AttentionMaskShapeReplacerTest, DoesNotReplaceNonBatchDim) {
    auto attention_mask = make_attention_mask();
    auto input_ids = std::make_shared<v0::Parameter>(element::i32, PartialShape{-1, -1});
    input_ids->set_friendly_name("input_ids");
    auto position_ids = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, 4, 1});
    position_ids->set_friendly_name("position_ids");
    position_ids->output(0).set_names({"position_ids"});
    // Only the batch dimension (index 0) is rewired; the sequence dimension is left untouched.
    model = build_model(attention_mask, attention_mask, input_ids, position_ids, {1});

    manager.register_pass<ov::pass::AttentionMaskShapeReplacer>(input_ids);

    auto attention_mask_ref = make_attention_mask();
    auto input_ids_ref = std::make_shared<v0::Parameter>(element::i32, PartialShape{-1, -1});
    input_ids_ref->set_friendly_name("input_ids");
    auto position_ids_ref = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, 4, 1});
    position_ids_ref->set_friendly_name("position_ids");
    position_ids_ref->output(0).set_names({"position_ids"});
    model_ref = build_model(attention_mask_ref, attention_mask_ref, input_ids_ref, position_ids_ref, {1});
}

TEST_F(AttentionMaskShapeReplacerTest, DoesNotReplaceNegativeIndex) {
    auto attention_mask = make_attention_mask();
    // A negative index would select a different dimension against the higher-rank source, so it is skipped.
    auto inputs_embeds = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, -1, 2048});
    inputs_embeds->set_friendly_name("inputs_embeds");
    auto position_ids = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, 4, 1});
    position_ids->set_friendly_name("position_ids");
    position_ids->output(0).set_names({"position_ids"});
    model = build_model(attention_mask, attention_mask, inputs_embeds, position_ids, {-2});

    manager.register_pass<ov::pass::AttentionMaskShapeReplacer>(inputs_embeds);

    auto attention_mask_ref = make_attention_mask();
    auto inputs_embeds_ref = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, -1, 2048});
    inputs_embeds_ref->set_friendly_name("inputs_embeds");
    auto position_ids_ref = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, 4, 1});
    position_ids_ref->set_friendly_name("position_ids");
    position_ids_ref->output(0).set_names({"position_ids"});
    model_ref = build_model(attention_mask_ref, attention_mask_ref, inputs_embeds_ref, position_ids_ref, {-2});
}
