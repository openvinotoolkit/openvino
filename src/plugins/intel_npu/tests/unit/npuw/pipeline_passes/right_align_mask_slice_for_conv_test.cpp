// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "npuw_transformations/right_align_mask_slice_for_conv.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/constant_folding.hpp"

namespace {

using namespace ov;

// Small hidden size to keep the constants (and hence the test) lightweight.
// The pass matches on op-types / connectivity, not on concrete sizes.
constexpr int64_t kHidden = 8;
constexpr int64_t kThreeHidden = 3 * kHidden;  // LFM2 in_proj produces B, C, x

// Appends a single LFM2 short-conv subgraph that consumes `attention_mask`, mirroring the
// real exported LFM2 IR (model.layers.N.conv). This exact chain repeats once per conv layer
// (10 times) all sharing the single `attention_mask` parameter:
//
//   attention_mask[-1,-1]
//        |  Slice: start=[0], stop=Reshape(Gather(ShapeOf(embeds),1,0),[1]), step=[1], axes=[1]
//        v  (LEFT-anchored -- the bug: NPUW right-pads input_ids, so [:, 0:seq] is all zeros)
//      Slice -> Unsqueeze -> Convert -> Multiply -> Add --+
//                                                         | (operand 1)
//                embeds ------------------------------> Multiply -> MatMul(in_proj) -> Transpose -> VariadicSplit(B,C,x)
//
// `results` and `params` are appended in place so several chains can share one mask.
std::shared_ptr<ov::op::v1::VariadicSplit> add_conv_mask_chain(
    const std::shared_ptr<ov::op::v0::Parameter>& attention_mask,
    int64_t seq_len,
    const std::string& tag,
    ov::ResultVector& results,
    ov::ParameterVector& params) {
    auto embeds = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, seq_len, kHidden});
    embeds->output(0).set_names({"embeds_" + tag});
    embeds->set_friendly_name("embeds_" + tag);
    params.push_back(embeds);

    // stop = Reshape(Gather(ShapeOf(embeds), idx=1, axis=0), [1]) == current seq length.
    auto shape_of = std::make_shared<ov::op::v3::ShapeOf>(embeds);
    auto gather_idx = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
    auto gather_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
    auto seq_len_scalar = std::make_shared<ov::op::v8::Gather>(shape_of, gather_idx, gather_axis);
    auto reshape_pattern = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto stop = std::make_shared<ov::op::v1::Reshape>(seq_len_scalar, reshape_pattern, false);
    stop->set_friendly_name("mask_slice_stop_" + tag);

    // attention_mask[:, 0:stop] on axis 1 (LEFT-anchored, as in the exported IR).
    auto start = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
    auto step = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto mask_slice = std::make_shared<ov::op::v8::Slice>(attention_mask, start, stop, step, axes);
    mask_slice->set_friendly_name("mask_slice_" + tag);

    auto unsqueeze_axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});
    auto mask_unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(mask_slice, unsqueeze_axes);
    mask_unsqueeze->set_friendly_name("mask_unsqueeze_" + tag);

    auto convert = std::make_shared<ov::op::v0::Convert>(mask_unsqueeze, ov::element::f32);
    convert->set_friendly_name("mask_convert_" + tag);

    // Real IR: Multiply({Convert, rsub Subtract}). Operand 0 must be the Convert.
    auto mul_constant = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 1, 1}, {1.0f});
    auto multiply = std::make_shared<ov::op::v1::Multiply>(convert, mul_constant);
    multiply->set_friendly_name("mask_multiply_" + tag);

    // Real IR: Add({Multiply, Convert}). Operand 0 must be the Multiply.
    auto add_constant = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 1, 1}, {0.0f});
    auto add = std::make_shared<ov::op::v1::Add>(multiply, add_constant);
    add->set_friendly_name("mask_add_" + tag);

    // Real IR: Multiply({operator_norm hidden states, Add}). `add` MUST be operand 1.
    auto masked = std::make_shared<ov::op::v1::Multiply>(embeds, add);
    masked->set_friendly_name("mask_masked_" + tag);

    auto in_proj_w = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{kHidden, kThreeHidden}, std::vector<float>(kHidden * kThreeHidden, 0.1f));
    auto matmul = std::make_shared<ov::op::v0::MatMul>(masked, in_proj_w);
    matmul->set_friendly_name("in_proj_" + tag);

    auto transpose_order = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {0, 2, 1});
    auto transpose = std::make_shared<ov::op::v1::Transpose>(matmul, transpose_order);
    transpose->set_friendly_name("transpose_" + tag);

    auto split_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
    auto split_lengths = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {kHidden, kHidden, kHidden});
    auto variadic_split = std::make_shared<ov::op::v1::VariadicSplit>(transpose, split_axis, split_lengths);
    variadic_split->set_friendly_name("variadic_split_" + tag);

    for (size_t i = 0; i < variadic_split->get_output_size(); ++i) {
        results.push_back(std::make_shared<ov::op::v0::Result>(variadic_split->output(i)));
    }
    return variadic_split;
}

// Builds a model with `num_conv_layers` conv-mask chains, all sharing one attention_mask.
std::shared_ptr<ov::Model> build_lfm2_conv_mask_model(const ov::PartialShape& mask_shape,
                                                      int64_t seq_len,
                                                      const std::string& mask_name = "attention_mask",
                                                      int num_conv_layers = 1) {
    auto attention_mask = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, mask_shape);
    attention_mask->output(0).set_names({mask_name});
    attention_mask->set_friendly_name(mask_name);

    ov::ResultVector results;
    ov::ParameterVector params{attention_mask};
    for (int i = 0; i < num_conv_layers; ++i) {
        add_conv_mask_chain(attention_mask, seq_len, std::to_string(i), results, params);
    }
    return std::make_shared<ov::Model>(results, params, "lfm2_conv_mask_model");
}

std::shared_ptr<ov::Node> get_node_by_name(const std::shared_ptr<ov::Model>& model, const std::string& name) {
    for (const auto& op : model->get_ops()) {
        if (op->get_friendly_name() == name) {
            return op;
        }
    }
    return nullptr;
}

// Returns the constant vector feeding `node`'s input `port`, after constant folding.
std::vector<int64_t> const_input_values(const std::shared_ptr<ov::Node>& node, size_t port) {
    auto producer = node->input_value(port).get_node_shared_ptr();
    auto constant = ov::as_type_ptr<ov::op::v0::Constant>(producer);
    if (!constant) {
        return {};
    }
    return constant->cast_vector<int64_t>();
}

// The conv mask Slice must be re-anchored to the *right* side of attention_mask.
// Before: attention_mask[:, 0:seq_len]. After: attention_mask[:, mask_len-seq_len : mask_len].
TEST(RightAlignMaskSliceForConvTest, RewiresUnsqueezeToNewRightAnchoredSlice) {
    auto model = build_lfm2_conv_mask_model(ov::PartialShape{-1, -1}, /*seq_len=*/64);

    auto original_slice = get_node_by_name(model, "mask_slice_0");
    ASSERT_NE(original_slice, nullptr);

    ASSERT_NO_THROW(ov::npuw::RightAlignMaskSliceForConv().run_on_model(model));

    auto unsqueeze = get_node_by_name(model, "mask_unsqueeze_0");
    ASSERT_NE(unsqueeze, nullptr);

    // Unsqueeze must now be fed by a *different* Slice than the original left-anchored one.
    auto new_slice = unsqueeze->input_value(0).get_node_shared_ptr();
    ASSERT_EQ(std::string(new_slice->get_type_name()), "Slice");
    EXPECT_NE(new_slice.get(), original_slice.get()) << "Unsqueeze should be rewired to a new Slice";

    // New Slice data input is still the attention_mask parameter.
    auto slice_data = new_slice->input_value(0).get_node_shared_ptr();
    ASSERT_EQ(std::string(slice_data->get_type_name()), "Parameter");
    EXPECT_TRUE(slice_data->output(0).get_names().count("attention_mask") > 0);

    // New Slice start (input 1) is the computed offset: Subtract(mask_len, current_len).
    auto start_producer = new_slice->input_value(1).get_node_shared_ptr();
    EXPECT_EQ(std::string(start_producer->get_type_name()), "Subtract")
        << "Right-aligned start should be mask_len - current_len";

    // New Slice stop (input 2) is derived from ShapeOf(attention_mask) via Gather.
    auto stop_producer = new_slice->input_value(2).get_node_shared_ptr();  // Unsqueeze(mask_len)
    bool reaches_shape_of = false;
    std::shared_ptr<ov::Node> walk = stop_producer;
    for (int depth = 0; depth < 4 && walk; ++depth) {
        if (std::string(walk->get_type_name()) == "ShapeOf") {
            reaches_shape_of = true;
            break;
        }
        if (walk->get_input_size() == 0) {
            break;
        }
        walk = walk->input_value(0).get_node_shared_ptr();
    }
    EXPECT_TRUE(reaches_shape_of) << "Slice stop should trace back to ShapeOf(attention_mask)";
}

// With a static mask length, the new slice bounds must fold to [mask_len-seq_len, mask_len].
TEST(RightAlignMaskSliceForConvTest, ComputesRightAlignedBounds) {
    constexpr int64_t mask_len = 128;
    constexpr int64_t seq_len = 64;
    auto model = build_lfm2_conv_mask_model(ov::PartialShape{1, mask_len}, seq_len);

    ov::npuw::RightAlignMaskSliceForConv().run_on_model(model);
    // Fold ShapeOf/Gather/Subtract so we can read concrete bounds.
    ov::pass::ConstantFolding().run_on_model(model);

    auto unsqueeze = get_node_by_name(model, "mask_unsqueeze_0");
    ASSERT_NE(unsqueeze, nullptr);
    auto new_slice = unsqueeze->input_value(0).get_node_shared_ptr();
    ASSERT_EQ(std::string(new_slice->get_type_name()), "Slice");

    const auto start = const_input_values(new_slice, 1);
    const auto stop = const_input_values(new_slice, 2);
    const auto step = const_input_values(new_slice, 3);
    const auto axis = const_input_values(new_slice, 4);

    ASSERT_EQ(start.size(), 1u);
    ASSERT_EQ(stop.size(), 1u);
    EXPECT_EQ(start[0], mask_len - seq_len) << "start must be mask_len - seq_len (right-aligned)";
    EXPECT_EQ(stop[0], mask_len) << "stop must be mask_len";
    ASSERT_EQ(step.size(), 1u);
    EXPECT_EQ(step[0], 1);
    ASSERT_EQ(axis.size(), 1u);
    EXPECT_EQ(axis[0], 1) << "slice must operate on the sequence axis";
}

// Every conv layer that consumes attention_mask must be re-anchored (LFM2 has 10 of them).
TEST(RightAlignMaskSliceForConvTest, RewiresAllConvLayers) {
    auto model = build_lfm2_conv_mask_model(ov::PartialShape{-1, -1}, /*seq_len=*/64,
                                            "attention_mask", /*num_conv_layers=*/3);

    ov::npuw::RightAlignMaskSliceForConv().run_on_model(model);

    for (const std::string tag : {"0", "1", "2"}) {
        auto unsqueeze = get_node_by_name(model, "mask_unsqueeze_" + tag);
        ASSERT_NE(unsqueeze, nullptr) << "missing unsqueeze for layer " << tag;
        auto new_slice = unsqueeze->input_value(0).get_node_shared_ptr();
        ASSERT_EQ(std::string(new_slice->get_type_name()), "Slice") << "layer " << tag << " not rewired";
        EXPECT_EQ(std::string(new_slice->input_value(1).get_node_shared_ptr()->get_type_name()), "Subtract")
            << "layer " << tag << " slice is not right-anchored";
    }
}

// The pass keys on the parameter being named "attention_mask": other parameters are ignored.
TEST(RightAlignMaskSliceForConvTest, NoOpWhenParameterNotAttentionMask) {
    auto model = build_lfm2_conv_mask_model(ov::PartialShape{-1, -1}, /*seq_len=*/64, "input_ids");

    auto original_slice = get_node_by_name(model, "mask_slice_0");
    ASSERT_NE(original_slice, nullptr);

    ASSERT_NO_THROW(ov::npuw::RightAlignMaskSliceForConv().run_on_model(model));

    // Unsqueeze must still be fed by the original (unchanged) left-anchored slice.
    auto unsqueeze = get_node_by_name(model, "mask_unsqueeze_0");
    ASSERT_NE(unsqueeze, nullptr);
    EXPECT_EQ(unsqueeze->input_value(0).get_node_shared_ptr().get(), original_slice.get())
        << "Pass must not touch chains whose parameter is not named 'attention_mask'";
}

// A model without the conv-mask pattern must be left untouched.
TEST(RightAlignMaskSliceForConvTest, NoOpWhenPatternAbsent) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, kHidden});
    input->output(0).set_names({"attention_mask"});
    auto result = std::make_shared<ov::op::v0::Result>(input);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input}, "no_conv_model");

    const size_t ops_before = model->get_ops().size();
    ASSERT_NO_THROW(ov::npuw::RightAlignMaskSliceForConv().run_on_model(model));
    EXPECT_EQ(model->get_ops().size(), ops_before) << "No ops should be added when the pattern is absent";
}

}  // namespace
