// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <openvino/opsets/opset13.hpp>
#include <transformations/cpu_opset/common/op/sdp.hpp>
#include <transformations/cpu_opset/common/pass/stateful_sdp_fusion.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <openvino/pass/manager.hpp>
#include <ie_core.hpp>

#include "common_test_utils/ov_test_utils.hpp"

using namespace testing;
using namespace ov::intel_cpu;
using namespace ov;

static std::shared_ptr<ov::Model> makeSDPA(const ov::PartialShape& inputShape, bool isRef = false, bool hasConvert = false) {
    auto q = std::make_shared<ov::op::v0::Parameter>(element::f32, inputShape);
    auto k = std::make_shared<ov::op::v0::Parameter>(element::f32, inputShape);
    auto v = std::make_shared<ov::op::v0::Parameter>(element::f32, inputShape);
    auto init = std::make_shared<ov::op::v0::Parameter>(element::f32, inputShape);
    auto var_k = std::make_shared<ov::op::util::Variable>(
        ov::op::util::VariableInfo{inputShape, element::f32, "pastk"});
    std::shared_ptr<ov::Node> pastk = std::make_shared<ov::op::v6::ReadValue>(k, var_k);
    auto var_v = std::make_shared<ov::op::util::Variable>(
        ov::op::util::VariableInfo{inputShape, element::f32, "pastv"});
    std::shared_ptr<ov::Node> pastv = std::make_shared<ov::op::v6::ReadValue>(v, var_v);
    Output<ov::Node> concatK, concatV, sdp;
    if (hasConvert) {
        pastk = std::make_shared<ov::op::v0::Convert>(pastk, element::f32);
        pastv = std::make_shared<ov::op::v0::Convert>(pastv, element::f32);
    }
    if (isRef) {
        ov::intel_cpu::ScaledDotProductAttentionStub::Config config;
        config.fuse_concat = true;
        auto new_node = std::make_shared<ov::intel_cpu::ScaledDotProductAttentionStub>(OutputVector{q, k, v, pastk, pastv}, config);
        sdp = new_node->output(0);
        concatK = new_node->output(1);
        concatV = new_node->output(2);
    } else {
        concatK = std::make_shared<ov::op::v0::Concat>(OutputVector{pastk, k}, 2);
        concatV = std::make_shared<ov::op::v0::Concat>(OutputVector{pastv, v}, 2);
        sdp = std::make_shared<ov::opset13::ScaledDotProductAttention>(q, concatK, concatV, false);
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
    return std::make_shared<Model>(results, sinks, ParameterVector{q, k, v, init}, "ConcatSDP");
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
            m.register_pass<StatefulSDPFusion>();
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
            m.register_pass<StatefulSDPFusion>();
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
