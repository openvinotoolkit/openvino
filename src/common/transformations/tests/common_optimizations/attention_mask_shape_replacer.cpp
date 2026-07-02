// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/paged_attention/attention_mask_shape_replacer.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
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

// data -> ShapeOf -> Gather(indices, axis=0) -> Result
std::shared_ptr<ov::Model> build_model(const Output<Node>& shape_source,
                                       const std::shared_ptr<v0::Parameter>& attention_mask,
                                       const std::shared_ptr<v0::Parameter>& input_source,
                                       const std::vector<int64_t>& gather_indices) {
    auto shape_of = std::make_shared<v3::ShapeOf>(shape_source, element::i64);
    auto indices = v0::Constant::create(element::i64, Shape{gather_indices.size()}, gather_indices);
    auto axis = v0::Constant::create(element::i64, Shape{}, {0});
    auto gather = std::make_shared<v8::Gather>(shape_of, indices, axis);
    auto result = std::make_shared<v0::Result>(gather);
    return std::make_shared<ov::Model>(ResultVector{result}, ParameterVector{attention_mask, input_source});
}

}  // namespace

class AttentionMaskShapeReplacerTest : public ::TransformationTestsF {};

TEST_F(AttentionMaskShapeReplacerTest, ReplacesWithInputIdsSequenceDim) {
    auto attention_mask = make_attention_mask();
    auto input_ids = std::make_shared<v0::Parameter>(element::i64, PartialShape{-1, -1});
    input_ids->set_friendly_name("input_ids");
    model = build_model(attention_mask, attention_mask, input_ids, {1});

    manager.register_pass<ov::pass::AttentionMaskShapeReplacer>(input_ids);

    auto attention_mask_ref = make_attention_mask();
    auto input_ids_ref = std::make_shared<v0::Parameter>(element::i64, PartialShape{-1, -1});
    input_ids_ref->set_friendly_name("input_ids");
    model_ref = build_model(input_ids_ref, attention_mask_ref, input_ids_ref, {1});
}

TEST_F(AttentionMaskShapeReplacerTest, ReplacesWithInputsEmbedsBothDims) {
    auto attention_mask = make_attention_mask();
    auto inputs_embeds = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, -1, 2048});
    inputs_embeds->set_friendly_name("inputs_embeds");
    model = build_model(attention_mask, attention_mask, inputs_embeds, {0, 1});

    manager.register_pass<ov::pass::AttentionMaskShapeReplacer>(inputs_embeds);

    auto attention_mask_ref = make_attention_mask();
    auto inputs_embeds_ref = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, -1, 2048});
    inputs_embeds_ref->set_friendly_name("inputs_embeds");
    model_ref = build_model(inputs_embeds_ref, attention_mask_ref, inputs_embeds_ref, {0, 1});
}

TEST_F(AttentionMaskShapeReplacerTest, DoesNotReplaceWhenIndexExceedsSourceRank) {
    auto attention_mask = make_attention_mask();
    // Rank-1 source cannot provide the sequence dimension at index 1.
    auto input_ids = std::make_shared<v0::Parameter>(element::i64, PartialShape{-1});
    input_ids->set_friendly_name("input_ids");
    model = build_model(attention_mask, attention_mask, input_ids, {1});

    manager.register_pass<ov::pass::AttentionMaskShapeReplacer>(input_ids);

    auto attention_mask_ref = make_attention_mask();
    auto input_ids_ref = std::make_shared<v0::Parameter>(element::i64, PartialShape{-1});
    input_ids_ref->set_friendly_name("input_ids");
    model_ref = build_model(attention_mask_ref, attention_mask_ref, input_ids_ref, {1});
}
