// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>
#include "openvino/opsets/opset13_decl.hpp"
#include <transformations/cpu_opset/common/op/sdpa.hpp>
#include <transformations/cpu_opset/common/pass/stateful_sdpa_fusion.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <transformations/utils/gen_pattern.hpp>
#include <openvino/pass/manager.hpp>
#include <ov_ops/type_relaxed.hpp>

#include "common_test_utils/ov_test_utils.hpp"
#include "transformations/utils/print_model.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/unsqueeze.hpp"

using namespace testing;
using namespace ov;
using namespace ov::intel_cpu;
using namespace ov::gen_pattern;

namespace {
    enum InsertPoint : uint8_t {
        At_None,
        At_Convert,
        At_Gather,
        At_MQ_Unsqueeze,
        At_MQ_Broadcast,
        At_MQ_Multiply,
        At_MQ_Reshape,
        At_End
    };
} // namespace

static std::shared_ptr<ov::Model> makeSDPA(const ov::PartialShape& inputShape, bool isRef = false, bool hasConvert = false, bool hasMultiquery = false,
    InsertPoint at = InsertPoint::At_None) {
    std::shared_ptr<ov::op::v0::Result> insert_result;
    auto q = std::make_shared<ov::op::v0::Parameter>(element::f32, inputShape);
    auto kvInputShape = inputShape;
    if (hasMultiquery) {
        kvInputShape[1] = inputShape[1] / 4;
    }
    auto k = std::make_shared<ov::op::v0::Parameter>(element::f32, kvInputShape);
    auto v = std::make_shared<ov::op::v0::Parameter>(element::f32, kvInputShape);
    auto init = std::make_shared<ov::op::v0::Parameter>(element::f32, inputShape);
    auto beam_idx = std::make_shared<ov::op::v0::Parameter>(element::i32, ov::PartialShape{-1});
    auto var_k = std::make_shared<ov::op::util::Variable>(
        ov::op::util::VariableInfo{kvInputShape, element::f32, "pastk"});
    std::shared_ptr<ov::Node> pastk = std::make_shared<ov::op::v6::ReadValue>(k, var_k);
    auto var_v = std::make_shared<ov::op::util::Variable>(
        ov::op::util::VariableInfo{kvInputShape, element::f32, "pastv"});
    std::shared_ptr<ov::Node> pastv = std::make_shared<ov::op::v6::ReadValue>(v, var_v);
    Output<ov::Node> concatK, concatV, sdp;
    if (hasConvert) {
        pastk = std::make_shared<ov::op::v0::Convert>(pastk, element::f32);
        pastv = std::make_shared<ov::op::v0::Convert>(pastv, element::f32);
        if (at == InsertPoint::At_Convert) {
            // insert one should be enough to cover the 'convert' case
            insert_result = std::make_shared<ov::op::v0::Result>(pastk);
        }
    }
    if (isRef) {
        ov::intel_cpu::ScaledDotProductAttentionWithKVCache::Config config;
        config.fuse_concat = true;
        auto new_node = std::make_shared<ov::intel_cpu::ScaledDotProductAttentionWithKVCache>(OutputVector{q, k, v, beam_idx, pastk, pastv}, config);
        sdp = new_node->output(0);
        concatK = new_node->output(1);
        concatV = new_node->output(2);
    } else {
        pastk = std::make_shared<ov::op::v8::Gather>(pastk, beam_idx, op::v0::Constant::create(element::i32, {1}, {0}));
        if (at == InsertPoint::At_Gather) {
            insert_result = std::make_shared<ov::op::v0::Result>(pastk);
        }
        pastv = std::make_shared<ov::op::v8::Gather>(pastv, beam_idx, op::v0::Constant::create(element::i32, {1}, {0}));
        concatK = std::make_shared<ov::op::v0::Concat>(OutputVector{pastk, k}, 2);
        concatV = std::make_shared<ov::op::v0::Concat>(OutputVector{pastv, v}, 2);
        if (hasMultiquery) {
            auto make_multi_query = [&] (const Output<Node>& conat) {
                auto beam_idx_shape = makeOP<ov::op::TypeRelaxed<opset1::ShapeOf>>({beam_idx},
                    {{"type_relax", true}, {"input_data_types", {}}, {"output_data_types", {element::i32}}});
                auto unsqueeze_concat = makeOP<opset1::Unsqueeze>({conat, 2});
                if (at == InsertPoint::At_MQ_Unsqueeze && !insert_result) {
                    insert_result = std::make_shared<ov::op::v0::Result>(unsqueeze_concat);
                }
                auto concat_shape = makeOP<ov::op::TypeRelaxed<opset1::ShapeOf>>(
                    {conat},
                    {{"type_relax", true}, {"input_data_types", {}}, {"output_data_types", {element::i32}}});
                auto gather_ls = makeOP<opset8::Gather>({concat_shape, {2, 3}, 0}, {{"batch_dims", 0}});
                auto expected_group_shape = makeOP<opset1::Concat>({beam_idx_shape, {inputShape[1] / 4}, {4}, gather_ls}, {{"axis", 0}});
                auto expand_Abs = makeOP<opset1::Abs>({expected_group_shape});
                auto axis_mapping = makeConst(element::u8, ov::Shape({}), {0});
                auto expand_ones = makeOP<opset1::Broadcast>({{1.0f},
                    expand_Abs,
                    axis_mapping}, {{"mode", "numpy"}});
                if (at == InsertPoint::At_MQ_Broadcast && !insert_result) {
                    insert_result = std::make_shared<ov::op::v0::Result>(expand_ones);
                }
                auto expand_Broadcast = makeOP<opset1::Multiply>({unsqueeze_concat,
                    expand_ones}, {{"auto_broadcast", "numpy"}});
                if (at == InsertPoint::At_MQ_Multiply && !insert_result) {
                    insert_result = std::make_shared<ov::op::v0::Result>(expand_Broadcast);
                }
                auto expected_shape = makeOP<opset1::Concat>({beam_idx_shape, {inputShape[1]}, gather_ls}, {{"axis", 0}});
                auto reshape_Reshape = makeOP<opset1::Reshape>({expand_Broadcast, expected_shape}, {{"special_zero", false}});
                if (at == InsertPoint::At_MQ_Reshape && !insert_result) {
                    insert_result = std::make_shared<ov::op::v0::Result>(reshape_Reshape);
                }
                return reshape_Reshape;
            };
            sdp = std::make_shared<ov::opset13::ScaledDotProductAttention>(q, make_multi_query(concatK), make_multi_query(concatV), false);
        } else {
            sdp = std::make_shared<ov::opset13::ScaledDotProductAttention>(q, concatK, concatV, false);
        }
    }
    if (hasConvert) {
        concatK = std::make_shared<ov::op::v0::Convert>(concatK, element::f32);
        concatV = std::make_shared<ov::op::v0::Convert>(concatV, element::f32);
    }
    auto pastk_assign = std::make_shared<op::v6::Assign>(concatK, var_k);
    auto pastv_assign = std::make_shared<op::v6::Assign>(concatV, var_v);
    auto add = std::make_shared<op::v1::Add>(sdp, op::v0::Constant::create(element::f32, {1}, {1.0f}));

    ResultVector results{std::make_shared<ov::op::v0::Result>(add)};
    if (insert_result)
        results.push_back(insert_result);
    SinkVector sinks{pastk_assign, pastv_assign};
    return std::make_shared<Model>(results, sinks, ParameterVector{q, k, v, init, beam_idx}, "ConcatSDP");
}

TEST(TransformationTests, StateConcatSDPA) {
#if defined(OPENVINO_ARCH_X86_64) && (defined(__ANDROID__) || defined(ANDROID))
    GTEST_SKIP() << "Skipping StateConcatSDPA test on Android X64";
#endif
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        using namespace ov;
        auto inputShape = ov::PartialShape{-1, 8, -1, 64};
        {
            f = makeSDPA(inputShape);
            pass::Manager m;
            m.register_pass<ov::pass::InitNodeInfo>();
            m.register_pass<StatefulSDPAFusion>();
            m.run_passes(f);
        }
        //construct ref interaction
        {
            f_ref = makeSDPA(inputShape, true);
        }
        auto res = compare_functions(f, f_ref);
        ASSERT_TRUE(res.first) << res.second;
    }
}

TEST(TransformationTests, StateConcatSDPAWithConvert) {
#if defined(OPENVINO_ARCH_X86_64) && (defined(__ANDROID__) || defined(ANDROID))
    GTEST_SKIP() << "Skipping StateConcatSDPAWithConvert test on Android X64";
#endif
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        using namespace ov;
        auto inputShape = ov::PartialShape{-1, 8, -1, 64};
        {
            f = makeSDPA(inputShape, false, true);
            pass::Manager m;
            m.register_pass<ov::pass::InitNodeInfo>();
            m.register_pass<StatefulSDPAFusion>();
            m.run_passes(f);
        }
        //construct ref interaction
        {
            f_ref = makeSDPA(inputShape, true, true);
        }
        auto res = compare_functions(f, f_ref);
        ASSERT_TRUE(res.first) << res.second;
    }
}

TEST(TransformationTests, StateConcatSDPAMixtral) {
#if defined(OPENVINO_ARCH_X86_64) && (defined(__ANDROID__) || defined(ANDROID))
    GTEST_SKIP() << "Skipping StateConcatSDPAMixtral test on Android X64";
#endif
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        using namespace ov;
        auto inputShape = ov::PartialShape{-1, 32, -1, 64};
        {
            f = makeSDPA(inputShape, false, false, true);
            pass::Manager m;
            m.register_pass<ov::pass::InitNodeInfo>();
            m.register_pass<StatefulSDPAFusion>();
            m.run_passes(f);
        }
        //construct ref interaction
        {
            f_ref = makeSDPA(inputShape, true, false, true);
        }
        auto res = compare_functions(f, f_ref);
        ASSERT_TRUE(res.first) << res.second;
    }
}

TEST(TransformationTests, StateConcatSDPAWithExtraNode) {
#if defined(OPENVINO_ARCH_X86_64) && (defined(__ANDROID__) || defined(ANDROID))
    GTEST_SKIP() << "Skipping StateConcatSDPAWithExtraNode test on Android X64";
#endif
    // when some unexpected extra nodes exist in SDPA, the fusion should fail
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        using namespace ov;
        auto inputShape = ov::PartialShape{-1, 32, -1, 64};
        size_t i = static_cast<size_t>(InsertPoint::At_None) + 1;
        size_t end = static_cast<size_t>(InsertPoint::At_End);
        // check each position
        for (; i < end; i++) {
            InsertPoint at = static_cast<InsertPoint>(i);
            f = makeSDPA(inputShape, false, true, true, at);
            f_ref = makeSDPA(inputShape, false, true, true, at);
            pass::Manager m;
            m.register_pass<ov::pass::InitNodeInfo>();
            m.register_pass<StatefulSDPAFusion>();
            m.run_passes(f);
            auto res = compare_functions(f, f_ref);
            ASSERT_TRUE(res.first) << res.second;
        }
    }
}

// Build a model with two SDPA blocks sharing the same KV-cache Variables.
// Path 1: ReadValue -> Gather -> Concat -> SDPA1
// Path 2: ReadValue -> Gather -> Concat -> SDPA2
// Both paths use the same var_k/var_v (shared KV-cache).
static std::shared_ptr<ov::Model> makeSharedKVModel(const ov::PartialShape& inputShape) {
    auto q1 = std::make_shared<ov::op::v0::Parameter>(element::f32, inputShape);
    auto k1 = std::make_shared<ov::op::v0::Parameter>(element::f32, inputShape);
    auto v1 = std::make_shared<ov::op::v0::Parameter>(element::f32, inputShape);
    auto q2 = std::make_shared<ov::op::v0::Parameter>(element::f32, inputShape);
    auto k2 = std::make_shared<ov::op::v0::Parameter>(element::f32, inputShape);
    auto v2 = std::make_shared<ov::op::v0::Parameter>(element::f32, inputShape);
    auto init_k = std::make_shared<ov::op::v0::Parameter>(element::f32, inputShape);
    auto init_v = std::make_shared<ov::op::v0::Parameter>(element::f32, inputShape);
    auto beam_idx = std::make_shared<ov::op::v0::Parameter>(element::i32, ov::PartialShape{-1});

    // Shared variables for KV-cache
    auto var_k = std::make_shared<ov::op::util::Variable>(
        ov::op::util::VariableInfo{inputShape, element::f32, "shared_pastk"});
    auto var_v = std::make_shared<ov::op::util::Variable>(
        ov::op::util::VariableInfo{inputShape, element::f32, "shared_pastv"});

    // Path 1: ReadValue -> Gather -> Concat -> SDPA1
    auto pastk1 = std::make_shared<ov::op::v6::ReadValue>(init_k, var_k);
    auto pastv1 = std::make_shared<ov::op::v6::ReadValue>(init_v, var_v);
    auto gather_k1 = std::make_shared<ov::op::v8::Gather>(
        pastk1, beam_idx, op::v0::Constant::create(element::i32, {1}, {0}));
    auto gather_v1 = std::make_shared<ov::op::v8::Gather>(
        pastv1, beam_idx, op::v0::Constant::create(element::i32, {1}, {0}));
    auto concat_k1 = std::make_shared<ov::op::v0::Concat>(OutputVector{gather_k1, k1}, 2);
    auto concat_v1 = std::make_shared<ov::op::v0::Concat>(OutputVector{gather_v1, v1}, 2);
    auto sdpa1 = std::make_shared<ov::opset13::ScaledDotProductAttention>(q1, concat_k1, concat_v1, false);
    auto add1 = std::make_shared<op::v1::Add>(sdpa1, op::v0::Constant::create(element::f32, {1}, {1.0f}));

    // Path 2: ReadValue -> Gather -> Concat -> SDPA2 (same variables, different ReadValue instances)
    auto pastk2 = std::make_shared<ov::op::v6::ReadValue>(init_k, var_k);
    auto pastv2 = std::make_shared<ov::op::v6::ReadValue>(init_v, var_v);
    auto gather_k2 = std::make_shared<ov::op::v8::Gather>(
        pastk2, beam_idx, op::v0::Constant::create(element::i32, {1}, {0}));
    auto gather_v2 = std::make_shared<ov::op::v8::Gather>(
        pastv2, beam_idx, op::v0::Constant::create(element::i32, {1}, {0}));
    auto concat_k2 = std::make_shared<ov::op::v0::Concat>(OutputVector{gather_k2, k2}, 2);
    auto concat_v2 = std::make_shared<ov::op::v0::Concat>(OutputVector{gather_v2, v2}, 2);
    auto sdpa2 = std::make_shared<ov::opset13::ScaledDotProductAttention>(q2, concat_k2, concat_v2, false);
    auto add2 = std::make_shared<op::v1::Add>(sdpa2, op::v0::Constant::create(element::f32, {1}, {1.0f}));

    // Assigns from path 1
    auto assign_k = std::make_shared<op::v6::Assign>(concat_k1, var_k);
    auto assign_v = std::make_shared<op::v6::Assign>(concat_v1, var_v);

    ResultVector results{std::make_shared<ov::op::v0::Result>(add1),
                         std::make_shared<ov::op::v0::Result>(add2)};
    SinkVector sinks{assign_k, assign_v};
    return std::make_shared<Model>(results, sinks,
        ParameterVector{q1, k1, v1, q2, k2, v2, init_k, init_v, beam_idx}, "SharedKVModel");
}

TEST(TransformationTests, StateConcatSDPASharedKVCache) {
#if defined(OPENVINO_ARCH_X86_64) && (defined(__ANDROID__) || defined(ANDROID))
    GTEST_SKIP() << "Skipping StateConcatSDPASharedKVCache test on Android X64";
#endif
    // When KV-cache is shared between multiple SDPA blocks, StatefulSDPAFusion must NOT apply.
    auto inputShape = ov::PartialShape{-1, 8, -1, 64};
    auto f = makeSharedKVModel(inputShape);
    auto f_ref = f->clone();

    ov::pass::Manager m;
    m.register_pass<ov::pass::InitNodeInfo>();
    m.register_pass<SDPASubgraphFusion>();
    m.run_passes(f);

    // Verify no ScaledDotProductAttentionWithKVCache was created (fusion was skipped)
    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, StateConcatSDPANonSharedViaSubgraphFusion) {
#if defined(OPENVINO_ARCH_X86_64) && (defined(__ANDROID__) || defined(ANDROID))
    GTEST_SKIP() << "Skipping StateConcatSDPANonSharedViaSubgraphFusion test on Android X64";
#endif
    // Non-shared KV-cache model must still be fused when running through SDPASubgraphFusion.
    auto inputShape = ov::PartialShape{-1, 8, -1, 64};
    auto f = makeSDPA(inputShape);
    auto f_ref = makeSDPA(inputShape, true);

    ov::pass::Manager m;
    m.register_pass<ov::pass::InitNodeInfo>();
    m.register_pass<SDPASubgraphFusion>();
    m.run_passes(f);

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}
