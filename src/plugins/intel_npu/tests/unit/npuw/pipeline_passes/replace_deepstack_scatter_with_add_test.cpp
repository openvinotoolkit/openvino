// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "npuw_transformations/replace_deepstack_scatter_with_add.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/gather_nd.hpp"
#include "openvino/op/non_zero.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/scatter_nd_update.hpp"
#include "openvino/op/transpose.hpp"

namespace {

constexpr int64_t kNumLayers = 3;
constexpr int64_t kSeq = 8;
constexpr int64_t kEmb = 4;

template <class Op>
std::size_t count_ops(const std::shared_ptr<ov::Model>& model) {
    std::size_t n = 0;
    for (const auto& op : model->get_ops()) {
        if (ov::is_type<Op>(op)) {
            ++n;
        }
    }
    return n;
}

std::shared_ptr<ov::op::v0::Parameter> find_param(const std::shared_ptr<ov::Model>& model, const std::string& needle) {
    for (const auto& param : model->get_parameters()) {
        if (param->get_friendly_name().find(needle) != std::string::npos) {
            return param;
        }
        for (const auto& name : param->output(0).get_names()) {
            if (name.find(needle) != std::string::npos) {
                return param;
            }
        }
    }
    return nullptr;
}

// Builds a minimal model reproducing the Qwen3-VL DeepStack injection: for each of
// kNumLayers levels, the hidden states are updated at visual-token positions via
//     ScatterNDUpdate(hidden, pos, GatherND(hidden, pos) + Gather(deepstack, L))
// where pos = Transpose(NonZero(visual_pos_masks)). Levels are chained.
//
// If `with_slice_assign` is true, each index_put ScatterNDUpdate is wrapped by a
// Reshape -> ScatterNDUpdate -> Reshape "SliceAssign" identity, matching how the
// PyTorch-traced model writes the whole tensor back.
std::shared_ptr<ov::Model> build_deepstack_model(bool with_slice_assign) {
    using ov::op::v0::Constant;
    using ov::op::v0::Parameter;
    using ov::op::v0::Result;

    auto hidden = std::make_shared<Parameter>(ov::element::f32, ov::PartialShape{1, kSeq, kEmb});
    hidden->set_friendly_name("inputs_embeds");
    hidden->output(0).set_names({"inputs_embeds"});

    auto deepstack = std::make_shared<Parameter>(ov::element::f32, ov::PartialShape{kNumLayers, -1, kEmb});
    deepstack->set_friendly_name("deepstack_visual_embeds");
    deepstack->output(0).set_names({"deepstack_visual_embeds"});

    auto vpm = std::make_shared<Parameter>(ov::element::boolean, ov::PartialShape{1, kSeq});
    vpm->set_friendly_name("visual_pos_masks");
    vpm->output(0).set_names({"visual_pos_masks"});

    auto nonzero = std::make_shared<ov::op::v3::NonZero>(vpm, ov::element::i64);
    auto perm = Constant::create(ov::element::i32, ov::Shape{2}, {1, 0});
    auto pos = std::make_shared<ov::op::v1::Transpose>(nonzero, perm);

    auto axis0 = Constant::create(ov::element::i64, ov::Shape{}, {0});

    ov::Output<ov::Node> cur = hidden;
    for (int64_t l = 0; l < kNumLayers; ++l) {
        auto gathered = std::make_shared<ov::op::v8::GatherND>(cur, pos);
        auto layer_idx = Constant::create(ov::element::i64, ov::Shape{}, {l});
        auto select = std::make_shared<ov::op::v8::Gather>(deepstack, layer_idx, axis0);
        auto add = std::make_shared<ov::op::v1::Add>(gathered, select);
        auto scatter = std::make_shared<ov::op::v3::ScatterNDUpdate>(cur, pos, add);

        if (with_slice_assign) {
            // Flatten -> ScatterNDUpdate over the whole tensor -> reshape back (identity).
            auto flat_shape = Constant::create(ov::element::i64, ov::Shape{1}, {kSeq * kEmb});
            auto reshape_in = std::make_shared<ov::op::v1::Reshape>(scatter, flat_shape, false);
            std::vector<int64_t> all_idx(kSeq * kEmb);
            for (int64_t i = 0; i < kSeq * kEmb; ++i) {
                all_idx[i] = i;
            }
            auto idx_const =
                Constant::create(ov::element::i64, ov::Shape{static_cast<size_t>(kSeq * kEmb), 1}, all_idx);
            auto base_flat = std::make_shared<ov::op::v1::Reshape>(cur, flat_shape, false);
            auto updates_flat = std::make_shared<ov::op::v1::Reshape>(scatter, flat_shape, false);
            auto slice_scatter = std::make_shared<ov::op::v3::ScatterNDUpdate>(base_flat, idx_const, updates_flat);
            auto out_shape = Constant::create(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{1, kSeq, kEmb});
            auto reshape_out = std::make_shared<ov::op::v1::Reshape>(slice_scatter, out_shape, false);
            cur = reshape_out;
        } else {
            cur = scatter;
        }
    }

    auto result = std::make_shared<Result>(cur);
    return std::make_shared<ov::Model>(ov::ResultVector{result},
                                       ov::ParameterVector{hidden, deepstack, vpm},
                                       "deepstack_test_model");
}

// --- Test 1 -------------------------------------------------------------------
// Core transformation: gather/scatter cluster replaced by residual adds, visual_pos_masks
// dropped, deepstack kept as a single [num_layers, ...] input.
TEST(ReplaceDeepstackScatterWithAddTest, ReplacesClusterAndDropsVisualPosMasks) {
    auto model = build_deepstack_model(/*with_slice_assign=*/false);

    ASSERT_EQ(count_ops<ov::op::v8::GatherND>(model), static_cast<std::size_t>(kNumLayers));
    ASSERT_EQ(count_ops<ov::op::v3::ScatterNDUpdate>(model), static_cast<std::size_t>(kNumLayers));
    ASSERT_EQ(count_ops<ov::op::v3::NonZero>(model), 1u);
    ASSERT_NE(find_param(model, "visual_pos_masks"), nullptr);
    ASSERT_EQ(model->get_parameters().size(), 3u);

    const bool changed = ov::npuw::ReplaceDeepstackScatterWithAdd().run_on_model(model);
    EXPECT_TRUE(changed);

    // The data-dependent ops are gone.
    EXPECT_EQ(count_ops<ov::op::v8::GatherND>(model), 0u);
    EXPECT_EQ(count_ops<ov::op::v3::ScatterNDUpdate>(model), 0u);
    EXPECT_EQ(count_ops<ov::op::v3::NonZero>(model), 0u);

    // visual_pos_masks input is dropped; deepstack is kept as a single [3, ...] input.
    EXPECT_EQ(find_param(model, "visual_pos_masks"), nullptr);
    EXPECT_EQ(model->get_parameters().size(), 2u);
    auto ds = find_param(model, "deepstack_visual_embeds");
    ASSERT_NE(ds, nullptr);
    ASSERT_EQ(ds->get_partial_shape().rank().get_length(), 3);
    EXPECT_EQ(ds->get_partial_shape()[0].get_length(), kNumLayers);

    // One residual Add per deepstack level remains, each fed by a select-Gather.
    EXPECT_EQ(count_ops<ov::op::v8::Gather>(model), static_cast<std::size_t>(kNumLayers));
    EXPECT_EQ(count_ops<ov::op::v1::Add>(model), static_cast<std::size_t>(kNumLayers));
}

// --- Test 2 -------------------------------------------------------------------
// Same, but with the SliceAssign identity wrapper present around each scatter.
TEST(ReplaceDeepstackScatterWithAddTest, BypassesSliceAssignWrapper) {
    auto model = build_deepstack_model(/*with_slice_assign=*/true);

    // 2 scatters per level: index_put + SliceAssign.
    ASSERT_EQ(count_ops<ov::op::v3::ScatterNDUpdate>(model), static_cast<std::size_t>(2 * kNumLayers));

    const bool changed = ov::npuw::ReplaceDeepstackScatterWithAdd().run_on_model(model);
    EXPECT_TRUE(changed);

    EXPECT_EQ(count_ops<ov::op::v8::GatherND>(model), 0u);
    EXPECT_EQ(count_ops<ov::op::v3::ScatterNDUpdate>(model), 0u);
    EXPECT_EQ(count_ops<ov::op::v3::NonZero>(model), 0u);
    EXPECT_EQ(find_param(model, "visual_pos_masks"), nullptr);
    EXPECT_EQ(model->get_parameters().size(), 2u);
    EXPECT_EQ(count_ops<ov::op::v1::Add>(model), static_cast<std::size_t>(kNumLayers));
}

// --- Test 3 -------------------------------------------------------------------
// A model without deepstack must be left untouched (pass returns false).
TEST(ReplaceDeepstackScatterWithAddTest, NoDeepstackIsUntouched) {
    using ov::op::v0::Parameter;
    using ov::op::v0::Result;

    auto hidden = std::make_shared<Parameter>(ov::element::f32, ov::PartialShape{1, kSeq, kEmb});
    hidden->set_friendly_name("inputs_embeds");
    auto bias = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 1, kEmb}, std::vector<float>(kEmb, 1.0f));
    auto add = std::make_shared<ov::op::v1::Add>(hidden, bias);
    auto result = std::make_shared<Result>(add);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{hidden}, "plain_model");

    const bool changed = ov::npuw::ReplaceDeepstackScatterWithAdd().run_on_model(model);

    EXPECT_FALSE(changed);
    EXPECT_EQ(model->get_parameters().size(), 1u);
    EXPECT_EQ(count_ops<ov::op::v1::Add>(model), 1u);
}

}  // namespace
