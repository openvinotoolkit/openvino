// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <openvino/opsets/opset13.hpp>
#include <transformations/cpu_opset/common/op/sdpa.hpp>
#include <transformations/cpu_opset/common/pass/stateful_sdpa_fusion.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <transformations/utils/gen_pattern.hpp>
#include <openvino/pass/manager.hpp>
#include <ov_ops/type_relaxed.hpp>

#include "common_test_utils/ov_test_utils.hpp"
#include "utils/print_model.hpp"

using namespace testing;
using namespace ov;
using namespace ov::intel_cpu;
using namespace ov::gen_pattern;

static std::shared_ptr<ov::Model> makeSDPA(const ov::PartialShape& inputShape, bool isRef = false, bool hasConvert = false, bool hasMultiquery = false) {
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
        pastv = std::make_shared<ov::op::v8::Gather>(pastv, beam_idx, op::v0::Constant::create(element::i32, {1}, {0}));
        concatK = std::make_shared<ov::op::v0::Concat>(OutputVector{pastk, k}, 2);
        concatV = std::make_shared<ov::op::v0::Concat>(OutputVector{pastv, v}, 2);
        if (hasMultiquery) {
            auto make_multi_query = [&] (const Output<Node>& conat) {
                auto beam_idx_shape = makeOP<ov::op::TypeRelaxed<opset1::ShapeOf>>({beam_idx},
                    {{"type_relax", true}, {"input_data_types", {}}, {"output_data_types", {element::i32}}});
                auto unsqueeze_concat = makeOP<opset1::Unsqueeze>({conat, 2});
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
                auto expand_Broadcast = makeOP<opset1::Multiply>({unsqueeze_concat,
                    expand_ones}, {{"auto_broadcast", "numpy"}});
                auto expected_shape = makeOP<opset1::Concat>({beam_idx_shape, {inputShape[1]}, gather_ls}, {{"axis", 0}});
                auto reshape_Reshape = makeOP<opset1::Reshape>({expand_Broadcast, expected_shape}, {{"special_zero", false}});
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
    SinkVector sinks{pastk_assign, pastv_assign};
    return std::make_shared<Model>(results, sinks, ParameterVector{q, k, v, init, beam_idx}, "ConcatSDP");
}

TEST(TransformationTests, StateConcatSDPA) {
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
