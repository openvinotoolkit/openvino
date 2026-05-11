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

    // Build a plain Gather→Concat→SDPA branch over (rv_k, rv_v).
    // Returned Concat outputs let the caller wire Assigns or ShapeOf consumers.
    struct PlainKvBranch {
        std::shared_ptr<ov::op::v13::ScaledDotProductAttention> sdpa;
        std::shared_ptr<ov::op::v0::Concat> concat_k;
        std::shared_ptr<ov::op::v0::Concat> concat_v;
    };

    inline PlainKvBranch make_plain_kv_branch(const std::shared_ptr<ov::Node>& q,
                                              const std::shared_ptr<ov::Node>& k_cur,
                                              const std::shared_ptr<ov::Node>& v_cur,
                                              const std::shared_ptr<ov::Node>& beam_idx,
                                              const std::shared_ptr<ov::Node>& rv_k,
                                              const std::shared_ptr<ov::Node>& rv_v) {
        auto axis = ov::op::v0::Constant::create(ov::element::i32, {1}, {0});
        auto g_k = std::make_shared<ov::op::v8::Gather>(rv_k, beam_idx, axis);
        auto g_v = std::make_shared<ov::op::v8::Gather>(rv_v, beam_idx, axis);
        auto c_k = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{g_k, k_cur}, 2);
        auto c_v = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{g_v, v_cur}, 2);
        auto sdp = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q, c_k, c_v, false);
        return {sdp, c_k, c_v};
    }

    // Build a fused ScaledDotProductAttentionWithKVCache branch — used in model_ref.
    inline std::shared_ptr<ov::intel_cpu::ScaledDotProductAttentionWithKVCache>
    make_fused_kv_branch(const std::shared_ptr<ov::Node>& q,
                         const std::shared_ptr<ov::Node>& k_cur,
                         const std::shared_ptr<ov::Node>& v_cur,
                         const std::shared_ptr<ov::Node>& beam_idx,
                         const std::shared_ptr<ov::Node>& rv_k,
                         const std::shared_ptr<ov::Node>& rv_v) {
        ov::intel_cpu::ScaledDotProductAttentionWithKVCache::Config config;
        config.fuse_concat = true;
        return std::make_shared<ov::intel_cpu::ScaledDotProductAttentionWithKVCache>(
            ov::OutputVector{q, k_cur, v_cur, beam_idx, rv_k, rv_v}, config);
    }
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
// One ReadValue per Variable fans out to two independent Gather -> Concat -> SDPA paths,
// mirroring the shared-KV-cache pattern seen in Gemma3n/Gemma4 exports.
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

    // Shared variables for KV-cache (single ReadValue per Variable, fanning out).
    auto var_k = std::make_shared<ov::op::util::Variable>(
        ov::op::util::VariableInfo{inputShape, element::f32, "shared_pastk"});
    auto var_v = std::make_shared<ov::op::util::Variable>(
        ov::op::util::VariableInfo{inputShape, element::f32, "shared_pastv"});
    auto pastk = std::make_shared<ov::op::v6::ReadValue>(init_k, var_k);
    auto pastv = std::make_shared<ov::op::v6::ReadValue>(init_v, var_v);

    // Path 1
    auto gather_k1 = std::make_shared<ov::op::v8::Gather>(
        pastk, beam_idx, op::v0::Constant::create(element::i32, {1}, {0}));
    auto gather_v1 = std::make_shared<ov::op::v8::Gather>(
        pastv, beam_idx, op::v0::Constant::create(element::i32, {1}, {0}));
    auto concat_k1 = std::make_shared<ov::op::v0::Concat>(OutputVector{gather_k1, k1}, 2);
    auto concat_v1 = std::make_shared<ov::op::v0::Concat>(OutputVector{gather_v1, v1}, 2);
    auto sdpa1 = std::make_shared<ov::opset13::ScaledDotProductAttention>(q1, concat_k1, concat_v1, false);
    auto add1 = std::make_shared<op::v1::Add>(sdpa1, op::v0::Constant::create(element::f32, {1}, {1.0f}));

    // Path 2 (fans out from the same ReadValue as path 1)
    auto gather_k2 = std::make_shared<ov::op::v8::Gather>(
        pastk, beam_idx, op::v0::Constant::create(element::i32, {1}, {0}));
    auto gather_v2 = std::make_shared<ov::op::v8::Gather>(
        pastv, beam_idx, op::v0::Constant::create(element::i32, {1}, {0}));
    auto concat_k2 = std::make_shared<ov::op::v0::Concat>(OutputVector{gather_k2, k2}, 2);
    auto concat_v2 = std::make_shared<ov::op::v0::Concat>(OutputVector{gather_v2, v2}, 2);
    auto sdpa2 = std::make_shared<ov::opset13::ScaledDotProductAttention>(q2, concat_k2, concat_v2, false);
    auto add2 = std::make_shared<op::v1::Add>(sdpa2, op::v0::Constant::create(element::f32, {1}, {1.0f}));

    // Single pair of Assigns for the shared Variables (from path 1's Concat).
    auto assign_k = std::make_shared<op::v6::Assign>(concat_k1, var_k);
    auto assign_v = std::make_shared<op::v6::Assign>(concat_v1, var_v);

    ResultVector results{std::make_shared<ov::op::v0::Result>(add1),
                         std::make_shared<ov::op::v0::Result>(add2)};
    SinkVector sinks{assign_k, assign_v};
    return std::make_shared<Model>(results, sinks,
        ParameterVector{q1, k1, v1, q2, k2, v2, init_k, init_v, beam_idx}, "SharedKVModel");
}

TEST_F(TransformationTestsF, StateConcatSDPASharedKVCache) {
#if defined(OPENVINO_ARCH_X86_64) && (defined(__ANDROID__) || defined(ANDROID))
    test_skipped = true;
    GTEST_SKIP() << "Skipping StateConcatSDPASharedKVCache test on Android X64";
#endif
    // When KV-cache is shared between multiple SDPA blocks, StatefulSDPAFusion must NOT apply:
    // leaving model_ref unset lets the fixture compare against a clone of the input model.
    auto inputShape = ov::PartialShape{-1, 8, -1, 64};
    model = makeSharedKVModel(inputShape);
    manager.register_pass<SDPASubgraphFusion>();
}

// Mix of SDPAs in one model:
//  - "shared" part: one pair of Variables feeds two SDPAs (SDPA_s1, SDPA_s2)
//  - "exclusive" part: another pair of Variables feeds a single SDPA (SDPA_e)
// After SDPASubgraphFusion, only SDPA_e must be fused to ScaledDotProductAttentionWithKVCache;
// SDPA_s1 and SDPA_s2 must remain as plain ScaledDotProductAttention.
static std::shared_ptr<ov::Model> makeMixedSharedAndExclusiveKVModel(const ov::PartialShape& inputShape,
                                                                     bool isRef = false) {
    auto make_param = [&](element::Type t, const ov::PartialShape& s) {
        return std::make_shared<ov::op::v0::Parameter>(t, s);
    };
    auto beam_idx = make_param(element::i32, ov::PartialShape{-1});

    // Shared part
    auto q_s1 = make_param(element::f32, inputShape);
    auto k_s1 = make_param(element::f32, inputShape);
    auto v_s1 = make_param(element::f32, inputShape);
    auto q_s2 = make_param(element::f32, inputShape);
    auto k_s2 = make_param(element::f32, inputShape);
    auto v_s2 = make_param(element::f32, inputShape);
    auto init_ks = make_param(element::f32, inputShape);
    auto init_vs = make_param(element::f32, inputShape);
    auto var_ks = std::make_shared<ov::op::util::Variable>(
        ov::op::util::VariableInfo{inputShape, element::f32, "shared_pastk"});
    auto var_vs = std::make_shared<ov::op::util::Variable>(
        ov::op::util::VariableInfo{inputShape, element::f32, "shared_pastv"});
    // Single ReadValue per shared Variable, fanning out to two SDPA paths.
    auto rv_ks = std::make_shared<ov::op::v6::ReadValue>(init_ks, var_ks);
    auto rv_vs = std::make_shared<ov::op::v6::ReadValue>(init_vs, var_vs);

    auto g_ks1 = std::make_shared<ov::op::v8::Gather>(rv_ks, beam_idx, op::v0::Constant::create(element::i32, {1}, {0}));
    auto g_vs1 = std::make_shared<ov::op::v8::Gather>(rv_vs, beam_idx, op::v0::Constant::create(element::i32, {1}, {0}));
    auto c_ks1 = std::make_shared<ov::op::v0::Concat>(OutputVector{g_ks1, k_s1}, 2);
    auto c_vs1 = std::make_shared<ov::op::v0::Concat>(OutputVector{g_vs1, v_s1}, 2);
    auto sdpa_s1 = std::make_shared<ov::opset13::ScaledDotProductAttention>(q_s1, c_ks1, c_vs1, false);
    auto add_s1 = std::make_shared<op::v1::Add>(sdpa_s1, op::v0::Constant::create(element::f32, {1}, {1.0f}));

    auto g_ks2 = std::make_shared<ov::op::v8::Gather>(rv_ks, beam_idx, op::v0::Constant::create(element::i32, {1}, {0}));
    auto g_vs2 = std::make_shared<ov::op::v8::Gather>(rv_vs, beam_idx, op::v0::Constant::create(element::i32, {1}, {0}));
    auto c_ks2 = std::make_shared<ov::op::v0::Concat>(OutputVector{g_ks2, k_s2}, 2);
    auto c_vs2 = std::make_shared<ov::op::v0::Concat>(OutputVector{g_vs2, v_s2}, 2);
    auto sdpa_s2 = std::make_shared<ov::opset13::ScaledDotProductAttention>(q_s2, c_ks2, c_vs2, false);
    auto add_s2 = std::make_shared<op::v1::Add>(sdpa_s2, op::v0::Constant::create(element::f32, {1}, {1.0f}));

    auto assign_ks = std::make_shared<op::v6::Assign>(c_ks1, var_ks);
    auto assign_vs = std::make_shared<op::v6::Assign>(c_vs1, var_vs);

    // Exclusive part (single SDPA on its own Variables — eligible for fusion)
    auto q_e = make_param(element::f32, inputShape);
    auto k_e = make_param(element::f32, inputShape);
    auto v_e = make_param(element::f32, inputShape);
    auto init_ke = make_param(element::f32, inputShape);
    auto init_ve = make_param(element::f32, inputShape);
    auto var_ke = std::make_shared<ov::op::util::Variable>(
        ov::op::util::VariableInfo{inputShape, element::f32, "excl_pastk"});
    auto var_ve = std::make_shared<ov::op::util::Variable>(
        ov::op::util::VariableInfo{inputShape, element::f32, "excl_pastv"});
    auto rv_ke = std::make_shared<ov::op::v6::ReadValue>(init_ke, var_ke);
    auto rv_ve = std::make_shared<ov::op::v6::ReadValue>(init_ve, var_ve);
    Output<ov::Node> sdp_e, concat_ke, concat_ve;
    if (isRef) {
        ov::intel_cpu::ScaledDotProductAttentionWithKVCache::Config config;
        config.fuse_concat = true;
        auto fused = std::make_shared<ov::intel_cpu::ScaledDotProductAttentionWithKVCache>(
            OutputVector{q_e, k_e, v_e, beam_idx, rv_ke, rv_ve}, config);
        sdp_e = fused->output(0);
        concat_ke = fused->output(1);
        concat_ve = fused->output(2);
    } else {
        auto g_ke =
            std::make_shared<ov::op::v8::Gather>(rv_ke, beam_idx, op::v0::Constant::create(element::i32, {1}, {0}));
        auto g_ve =
            std::make_shared<ov::op::v8::Gather>(rv_ve, beam_idx, op::v0::Constant::create(element::i32, {1}, {0}));
        concat_ke = std::make_shared<ov::op::v0::Concat>(OutputVector{g_ke, k_e}, 2);
        concat_ve = std::make_shared<ov::op::v0::Concat>(OutputVector{g_ve, v_e}, 2);
        sdp_e = std::make_shared<ov::opset13::ScaledDotProductAttention>(q_e, concat_ke, concat_ve, false);
    }
    auto add_e = std::make_shared<op::v1::Add>(sdp_e, op::v0::Constant::create(element::f32, {1}, {1.0f}));
    auto assign_ke = std::make_shared<op::v6::Assign>(concat_ke, var_ke);
    auto assign_ve = std::make_shared<op::v6::Assign>(concat_ve, var_ve);

    ResultVector results{std::make_shared<ov::op::v0::Result>(add_s1),
                         std::make_shared<ov::op::v0::Result>(add_s2),
                         std::make_shared<ov::op::v0::Result>(add_e)};
    SinkVector sinks{assign_ks, assign_vs, assign_ke, assign_ve};
    ParameterVector params{q_s1, k_s1, v_s1, q_s2, k_s2, v_s2, init_ks, init_vs,
                           q_e,  k_e,  v_e,  init_ke, init_ve, beam_idx};
    return std::make_shared<Model>(results, sinks, params, "MixedSharedExclusiveKV");
}

TEST_F(TransformationTestsF, StateConcatSDPAMixedSharedAndExclusive) {
#if defined(OPENVINO_ARCH_X86_64) && (defined(__ANDROID__) || defined(ANDROID))
    test_skipped = true;
    GTEST_SKIP() << "Skipping StateConcatSDPAMixedSharedAndExclusive test on Android X64";
#endif
    // Exclusive-cache SDPA must still be fused; shared-cache SDPAs must be left alone.
    auto inputShape = ov::PartialShape{-1, 8, -1, 64};
    model = makeMixedSharedAndExclusiveKVModel(inputShape);
    model_ref = makeMixedSharedAndExclusiveKVModel(inputShape, true);
    manager.register_pass<SDPASubgraphFusion>();
}

// Layer-0's Gather output feeds a ShapeOf chain shared with both SDPAs' attention-mask input
// (Qwen3-style RoPE fanout). Both SDPAs must still fuse — shape metadata is not shared cache.
// SDPA counts checked manually: SimplifyGatherShapeOf inside SDPASubgraphFusion mutates the
// ShapeOf chain in ways that make a precise post-pass model_ref fragile.
static std::shared_ptr<ov::Model>
makeNonSharedKVWithSharedShapeOfRopeModel(const ov::PartialShape& inputShape) {
    auto make_param = [&](ov::element::Type t, const ov::PartialShape& s) {
        return std::make_shared<ov::op::v0::Parameter>(t, s);
    };
    auto make_var = [&](const std::string& id) {
        return std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{inputShape, ov::element::f32, id});
    };
    auto beam_idx = make_param(ov::element::i32, ov::PartialShape{-1});

    // Layer 0
    auto q0 = make_param(ov::element::f32, inputShape);
    auto k0 = make_param(ov::element::f32, inputShape);
    auto v0 = make_param(ov::element::f32, inputShape);
    auto init_k0 = make_param(ov::element::f32, inputShape);
    auto init_v0 = make_param(ov::element::f32, inputShape);
    auto var_k0 = make_var("pastk_0");
    auto var_v0 = make_var("pastv_0");
    auto rv_k0 = std::make_shared<ov::op::v6::ReadValue>(init_k0, var_k0);
    auto rv_v0 = std::make_shared<ov::op::v6::ReadValue>(init_v0, var_v0);
    auto axis = ov::op::v0::Constant::create(ov::element::i32, {1}, {0});
    auto g_k0 = std::make_shared<ov::op::v8::Gather>(rv_k0, beam_idx, axis);
    auto g_v0 = std::make_shared<ov::op::v8::Gather>(rv_v0, beam_idx, axis);
    auto c_k0 = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{g_k0, k0}, 2);
    auto c_v0 = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{g_v0, v0}, 2);

    // Shared shape/metadata subgraph rooted on layer 0's Gather output.
    auto shape_of = std::make_shared<ov::op::v3::ShapeOf>(g_k0, ov::element::i64);
    auto seq_len =
        std::make_shared<ov::op::v8::Gather>(shape_of,
                                             ov::op::v0::Constant::create(ov::element::i64, {1}, {2}),
                                             ov::op::v0::Constant::create(ov::element::i64, {}, {0}));
    auto pos_shape = std::make_shared<ov::op::v0::Concat>(
        ov::OutputVector{ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                         ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                         ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                         seq_len},
        0);
    // Non-zero constant prevents NopElimination from collapsing the
    // downstream Add nodes into identities, which would erase the shared
    // fanout entirely.
    auto pos_const = ov::op::v0::Constant::create(ov::element::f32, {1, 1, 1, 1}, {0.7f});
    auto pos_bcast = std::make_shared<ov::op::v3::Broadcast>(pos_const, pos_shape);

    auto mask0_param = make_param(ov::element::f32, ov::PartialShape{-1, 8, -1, -1});
    auto mask1_param = make_param(ov::element::f32, ov::PartialShape{-1, 8, -1, -1});
    auto mask0 = std::make_shared<ov::op::v1::Add>(mask0_param, pos_bcast);
    auto mask1 = std::make_shared<ov::op::v1::Add>(mask1_param, pos_bcast);

    auto sdp0 = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q0, c_k0, c_v0, mask0, false);
    auto add0 = std::make_shared<ov::op::v1::Add>(sdp0,
                                                  ov::op::v0::Constant::create(ov::element::f32, {1}, {1.0f}));
    auto assign_k0 = std::make_shared<ov::op::v6::Assign>(c_k0, var_k0);
    auto assign_v0 = std::make_shared<ov::op::v6::Assign>(c_v0, var_v0);

    // Layer 1 — independent Variables, no cache sharing with layer 0
    auto q1 = make_param(ov::element::f32, inputShape);
    auto k1 = make_param(ov::element::f32, inputShape);
    auto v1 = make_param(ov::element::f32, inputShape);
    auto init_k1 = make_param(ov::element::f32, inputShape);
    auto init_v1 = make_param(ov::element::f32, inputShape);
    auto var_k1 = make_var("pastk_1");
    auto var_v1 = make_var("pastv_1");
    auto rv_k1 = std::make_shared<ov::op::v6::ReadValue>(init_k1, var_k1);
    auto rv_v1 = std::make_shared<ov::op::v6::ReadValue>(init_v1, var_v1);
    auto branch1 = make_plain_kv_branch(q1, k1, v1, beam_idx, rv_k1, rv_v1);
    // Replace branch1.sdpa with a masked variant to mirror layer 0's mask wiring.
    auto sdp1 = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q1,
                                                                         branch1.concat_k,
                                                                         branch1.concat_v,
                                                                         mask1,
                                                                         false);
    auto add1 = std::make_shared<ov::op::v1::Add>(sdp1,
                                                  ov::op::v0::Constant::create(ov::element::f32, {1}, {1.0f}));
    auto assign_k1 = std::make_shared<ov::op::v6::Assign>(branch1.concat_k, var_k1);
    auto assign_v1 = std::make_shared<ov::op::v6::Assign>(branch1.concat_v, var_v1);

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(add0),
                             std::make_shared<ov::op::v0::Result>(add1)};
    ov::SinkVector sinks{assign_k0, assign_v0, assign_k1, assign_v1};
    ov::ParameterVector params{q0, k0, v0, init_k0, init_v0, mask0_param, q1, k1,
                               v1, init_k1, init_v1, mask1_param, beam_idx};
    return std::make_shared<ov::Model>(results, sinks, params, "NonSharedKVWithSharedShapeOfRope");
}

TEST(StateConcatSDPAExtra, NonSharedKVWithSharedShapeOfRope) {
#if defined(OPENVINO_ARCH_X86_64) && (defined(__ANDROID__) || defined(ANDROID))
    GTEST_SKIP() << "Skipping NonSharedKVWithSharedShapeOfRope on Android X64";
#endif
    auto inputShape = ov::PartialShape{-1, 8, -1, 64};
    auto model = makeNonSharedKVWithSharedShapeOfRopeModel(inputShape);

    ov::pass::Manager manager;
    manager.register_pass<SDPASubgraphFusion>();
    manager.run_passes(model);

    size_t plain_sdpa = 0;
    size_t fused_sdpa = 0;
    for (const auto& op : model->get_ordered_ops()) {
        if (ov::is_type<ov::op::v13::ScaledDotProductAttention>(op)) {
            ++plain_sdpa;
        }
        if (ov::is_type<ov::intel_cpu::ScaledDotProductAttentionWithKVCache>(op)) {
            ++fused_sdpa;
        }
    }
    EXPECT_EQ(plain_sdpa, 0u)
        << "Both SDPAs must be fused despite a shared ShapeOf-rooted fanout into attention-mask inputs.";
    EXPECT_EQ(fused_sdpa, 2u) << "Both SDPAs must be fused into ScaledDotProductAttentionWithKVCache.";
}

// One (RV, SDPA) pair with a ShapeOf reading Concat_K (Gemma3n layer-0 topology).
// Fusion must apply and the post-fusion reroute must leave ShapeOf reading from the fused K output.
static std::shared_ptr<ov::Model>
makeSingleSDPAWithShapeOfOnConcatKModel(const ov::PartialShape& inputShape, bool isRef = false) {
    auto q = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputShape);
    auto k = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputShape);
    auto v = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputShape);
    auto init_k = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputShape);
    auto init_v = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputShape);
    auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});

    auto var_k = std::make_shared<ov::op::util::Variable>(
        ov::op::util::VariableInfo{inputShape, ov::element::f32, "pastk"});
    auto var_v = std::make_shared<ov::op::util::Variable>(
        ov::op::util::VariableInfo{inputShape, ov::element::f32, "pastv"});
    auto past_k = std::make_shared<ov::op::v6::ReadValue>(init_k, var_k);
    auto past_v = std::make_shared<ov::op::v6::ReadValue>(init_v, var_v);

    ov::Output<ov::Node> sdp_out;
    ov::Output<ov::Node> shape_of_input;  // ShapeOf reads K-cache shape from here.
    std::shared_ptr<ov::op::v6::Assign> assign_k_node;
    std::shared_ptr<ov::op::v6::Assign> assign_v_node;
    if (isRef) {
        auto fused = make_fused_kv_branch(q, k, v, beam_idx, past_k, past_v);
        sdp_out = fused->output(0);
        // ShapeOf re-routed onto the fused node's K-output by the matcher.
        shape_of_input = fused->output(1);
        assign_k_node = std::make_shared<ov::op::v6::Assign>(fused->output(1), var_k);
        assign_v_node = std::make_shared<ov::op::v6::Assign>(fused->output(2), var_v);
    } else {
        auto branch = make_plain_kv_branch(q, k, v, beam_idx, past_k, past_v);
        sdp_out = branch.sdpa->output(0);
        shape_of_input = branch.concat_k->output(0);
        assign_k_node = std::make_shared<ov::op::v6::Assign>(branch.concat_k, var_k);
        assign_v_node = std::make_shared<ov::op::v6::Assign>(branch.concat_v, var_v);
    }
    auto shape_of_ck = std::make_shared<ov::op::v3::ShapeOf>(shape_of_input, ov::element::i64);

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(sdp_out),
                             std::make_shared<ov::op::v0::Result>(shape_of_ck)};
    ov::SinkVector sinks{assign_k_node, assign_v_node};
    return std::make_shared<ov::Model>(results,
                                       sinks,
                                       ov::ParameterVector{q, k, v, init_k, init_v, beam_idx},
                                       "SingleSDPAWithShapeOfOnConcatK");
}

TEST_F(TransformationTestsF, StateConcatSDPAShapeOfOnConcatKRerouteSucceeds) {
#if defined(OPENVINO_ARCH_X86_64) && (defined(__ANDROID__) || defined(ANDROID))
    test_skipped = true;
    GTEST_SKIP() << "Skipping ShapeOfOnConcatKRerouteSucceeds on Android X64";
#endif
    auto inputShape = ov::PartialShape{-1, 8, -1, 64};
    model = makeSingleSDPAWithShapeOfOnConcatKModel(inputShape);
    model_ref = makeSingleSDPAWithShapeOfOnConcatKModel(inputShape, /*isRef=*/true);
    manager.register_pass<StatefulSDPAFusion>();
}

// Gemma3n shared-KV topology: one ReadValue fans out to two SDPAs.
// Writer (topologically first) fuses; reader stays plain.
static std::shared_ptr<ov::Model>
makeGemma3nLikeSharedKVModel(const ov::PartialShape& inputShape, bool isRef = false) {
    auto q1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputShape);
    auto k1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputShape);
    auto v1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputShape);
    auto q2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputShape);
    auto k2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputShape);
    auto v2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputShape);
    auto init_k = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputShape);
    auto init_v = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputShape);
    auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});

    auto var_k = std::make_shared<ov::op::util::Variable>(
        ov::op::util::VariableInfo{inputShape, ov::element::f32, "shared_pastk"});
    auto var_v = std::make_shared<ov::op::util::Variable>(
        ov::op::util::VariableInfo{inputShape, ov::element::f32, "shared_pastv"});
    auto past_k = std::make_shared<ov::op::v6::ReadValue>(init_k, var_k);
    auto past_v = std::make_shared<ov::op::v6::ReadValue>(init_v, var_v);

    ov::Output<ov::Node> sdp1_out, sdp2_out;
    std::shared_ptr<ov::op::v6::Assign> assign_k, assign_v;
    if (isRef) {
        // Branch 1 = writer → fused. Its K / V outputs feed the shared Assigns.
        auto fused = make_fused_kv_branch(q1, k1, v1, beam_idx, past_k, past_v);
        sdp1_out = fused->output(0);
        assign_k = std::make_shared<ov::op::v6::Assign>(fused->output(1), var_k);
        assign_v = std::make_shared<ov::op::v6::Assign>(fused->output(2), var_v);
        // Branch 2 = reader → must NOT fuse.
        auto reader = make_plain_kv_branch(q2, k2, v2, beam_idx, past_k, past_v);
        sdp2_out = reader.sdpa->output(0);
    } else {
        auto branch1 = make_plain_kv_branch(q1, k1, v1, beam_idx, past_k, past_v);
        auto branch2 = make_plain_kv_branch(q2, k2, v2, beam_idx, past_k, past_v);
        sdp1_out = branch1.sdpa->output(0);
        sdp2_out = branch2.sdpa->output(0);
        // One Assign pair writes back to the shared Variables (from branch 1).
        assign_k = std::make_shared<ov::op::v6::Assign>(branch1.concat_k, var_k);
        assign_v = std::make_shared<ov::op::v6::Assign>(branch1.concat_v, var_v);
    }

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(sdp1_out),
                             std::make_shared<ov::op::v0::Result>(sdp2_out)};
    ov::SinkVector sinks{assign_k, assign_v};
    return std::make_shared<ov::Model>(results,
                                       sinks,
                                       ov::ParameterVector{q1, k1, v1, q2, k2, v2, init_k, init_v, beam_idx},
                                       "Gemma3nLikeSharedKV");
}

TEST_F(TransformationTestsF, StateConcatSDPAGemma3nLikeSharedKV) {
#if defined(OPENVINO_ARCH_X86_64) && (defined(__ANDROID__) || defined(ANDROID))
    test_skipped = true;
    GTEST_SKIP() << "Skipping Gemma3nLikeSharedKV on Android X64";
#endif
    auto inputShape = ov::PartialShape{-1, 8, -1, 64};
    model = makeGemma3nLikeSharedKVModel(inputShape);
    model_ref = makeGemma3nLikeSharedKVModel(inputShape, /*isRef=*/true);
    manager.register_pass<StatefulSDPAFusion>();
}

// Two ReadValue nodes aliasing one Variable: writer fuses, reader stays plain.
// SDPA counts checked manually — graph_comparator pairs sinks by variable_id only, and 4 Assigns
// sharing 2 variable_ids cannot be paired unambiguously.
static std::shared_ptr<ov::Model>
makeSharedVariableTwoReadValuesModel(const ov::PartialShape& inputShape) {
    auto q1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputShape);
    auto k1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputShape);
    auto v1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputShape);
    auto q2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputShape);
    auto k2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputShape);
    auto v2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputShape);
    auto init_k1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputShape);
    auto init_v1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputShape);
    auto init_k2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputShape);
    auto init_v2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputShape);
    auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});

    // ONE Variable per K and V — but TWO ReadValue/Assign pairs each.
    auto var_k = std::make_shared<ov::op::util::Variable>(
        ov::op::util::VariableInfo{inputShape, ov::element::f32, "shared_var_k"});
    auto var_v = std::make_shared<ov::op::util::Variable>(
        ov::op::util::VariableInfo{inputShape, ov::element::f32, "shared_var_v"});

    auto rv_k1 = std::make_shared<ov::op::v6::ReadValue>(init_k1, var_k);
    auto rv_v1 = std::make_shared<ov::op::v6::ReadValue>(init_v1, var_v);
    auto rv_k2 = std::make_shared<ov::op::v6::ReadValue>(init_k2, var_k);
    auto rv_v2 = std::make_shared<ov::op::v6::ReadValue>(init_v2, var_v);

    auto branch1 = make_plain_kv_branch(q1, k1, v1, beam_idx, rv_k1, rv_v1);
    auto branch2 = make_plain_kv_branch(q2, k2, v2, beam_idx, rv_k2, rv_v2);
    auto assign_k1 = std::make_shared<ov::op::v6::Assign>(branch1.concat_k, var_k);
    auto assign_v1 = std::make_shared<ov::op::v6::Assign>(branch1.concat_v, var_v);
    auto assign_k2 = std::make_shared<ov::op::v6::Assign>(branch2.concat_k, var_k);
    auto assign_v2 = std::make_shared<ov::op::v6::Assign>(branch2.concat_v, var_v);

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(branch1.sdpa),
                             std::make_shared<ov::op::v0::Result>(branch2.sdpa)};
    ov::SinkVector sinks{assign_k1, assign_v1, assign_k2, assign_v2};
    return std::make_shared<ov::Model>(
        results,
        sinks,
        ov::ParameterVector{q1, k1, v1, q2, k2, v2, init_k1, init_v1, init_k2, init_v2, beam_idx},
        "SharedVariableTwoReadValues");
}

TEST(StateConcatSDPAExtra, SharedVariableTwoReadValuesPartiallyFuses) {
#if defined(OPENVINO_ARCH_X86_64) && (defined(__ANDROID__) || defined(ANDROID))
    GTEST_SKIP() << "Skipping SharedVariableTwoReadValuesPartiallyFuses on Android X64";
#endif
    auto inputShape = ov::PartialShape{-1, 8, -1, 64};
    auto model = makeSharedVariableTwoReadValuesModel(inputShape);

    ov::pass::Manager manager;
    manager.register_pass<StatefulSDPAFusion>();
    manager.run_passes(model);

    size_t plain_sdpa = 0;
    size_t fused_sdpa = 0;
    for (const auto& op : model->get_ordered_ops()) {
        if (ov::is_type<ov::op::v13::ScaledDotProductAttention>(op)) {
            ++plain_sdpa;
        }
        if (ov::is_type<ov::intel_cpu::ScaledDotProductAttentionWithKVCache>(op)) {
            ++fused_sdpa;
        }
    }
    EXPECT_EQ(plain_sdpa, 1u)
        << "Variable-id-keyed detection must leave the reader SDPA as plain "
           "v13::ScaledDotProductAttention even though its ReadValue node is "
           "distinct from the writer's.";
    EXPECT_EQ(fused_sdpa, 1u)
        << "The writer (topologically-first) SDPA on the shared Variable must fuse.";
}

