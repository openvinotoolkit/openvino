// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/cpu_opset/x64/pass/mlp_fusion.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/common_optimizations/activations_scaling.hpp"
#include "transformations/cpu_opset/x64/op/llm_mlp.hpp"

using namespace ov;
using namespace ov::op;
using ov::intel_cpu::LLMMLPNode;

namespace {
constexpr size_t kHidden = 16;
constexpr size_t kUp = 32;
constexpr float kScale = 8.0F;

using GateUpType = LLMMLPNode::GATE_UP_TYPE;

std::shared_ptr<v0::Constant> make_weights(size_t rows,
                                           size_t cols,
                                           float start,
                                           size_t scaled_begin = 0,
                                           size_t scaled_end = 0) {
    std::vector<float> values(rows * cols);
    for (size_t i = 0; i < values.size(); ++i) {
        const bool scaled = i / cols >= scaled_begin && i / cols < scaled_end;
        values[i] = (start + 0.001F * static_cast<float>(i % 97)) * (scaled ? kScale : 1.0F);
    }
    return v0::Constant::create(element::f16, Shape{rows, cols}, values);
}

std::shared_ptr<Model> make_mlp(GateUpType type, bool scaled_gate) {
    auto input = std::make_shared<v0::Parameter>(element::f16, PartialShape{-1, -1, kHidden});
    Output<Node> gate;
    Output<Node> up;
    if (type == GateUpType::SEPARATE) {
        gate = std::make_shared<v0::MatMul>(input, make_weights(kUp, kHidden, 0.1F), false, true);
        up = std::make_shared<v0::MatMul>(input, make_weights(kUp, kHidden, 0.2F), false, true);
    } else {
        auto gate_up = std::make_shared<v0::MatMul>(input, make_weights(2 * kUp, kHidden, 0.1F), false, true);
        auto split = std::make_shared<v1::VariadicSplit>(gate_up,
                                                         v0::Constant::create(element::i32, Shape{}, {-1}),
                                                         v0::Constant::create(element::i32, Shape{2}, {kUp, kUp}));
        const bool gate_first = type == GateUpType::COMBINED_GATE_UP;
        gate = split->output(gate_first ? 0 : 1);
        up = split->output(gate_first ? 1 : 0);
    }
    if (scaled_gate) {
        gate = std::make_shared<v1::Multiply>(gate, v0::Constant::create(element::f16, Shape{}, {kScale}));
    }
    auto gated = std::make_shared<v1::Multiply>(std::make_shared<v4::Swish>(gate), up);
    auto down = std::make_shared<v0::MatMul>(gated, make_weights(kHidden, kUp, 0.3F), false, true);
    return std::make_shared<Model>(OutputVector{down}, ParameterVector{input});
}

void register_mlp_fusion(ov::pass::Manager& manager) {
    manager.register_pass<ov::intel_cpu::MLPFusion>();
    manager.get_pass_config()->set_callback<ov::intel_cpu::MLPFusionPass>([](const std::shared_ptr<const Node>&) {
        return true;
    });
}

std::string type_name(const testing::TestParamInfo<GateUpType>& info) {
    switch (info.param) {
    case GateUpType::SEPARATE:
        return "Separate";
    case GateUpType::COMBINED_GATE_UP:
        return "CombinedGateUp";
    default:
        return "CombinedUpGate";
    }
}

class MLPFusionGateScaleTest : public TransformationTestsF, public testing::WithParamInterface<GateUpType> {};
class MLPFusionAfterActivationsScalingTest : public testing::TestWithParam<GateUpType> {};
}  // namespace

TEST_P(MLPFusionGateScaleTest, FoldedIntoGateRows) {
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    disable_rt_info_check();
    const auto type = GetParam();
    model = make_mlp(type, true);
    register_mlp_fusion(manager);
    {
        auto input = std::make_shared<v0::Parameter>(element::f16, PartialShape{-1, -1, kHidden});
        std::shared_ptr<v0::Constant> gate_w;
        std::shared_ptr<v0::Constant> up_w;
        if (type == GateUpType::SEPARATE) {
            gate_w = make_weights(kUp, kHidden, 0.1F, 0, kUp);
            up_w = make_weights(kUp, kHidden, 0.2F);
        } else {
            const size_t gate_begin = type == GateUpType::COMBINED_GATE_UP ? 0 : kUp;
            gate_w = make_weights(2 * kUp, kHidden, 0.1F, gate_begin, gate_begin + kUp);
            up_w = gate_w;
        }
        LLMMLPNode::Config config{};
        config.act = LLMMLPNode::ACT_FN::SILU;
        config.hidden_size = static_cast<int>(kHidden);
        config.up_size = static_cast<int>(kUp);
        config.gate_up_type = type;
        auto mlp =
            std::make_shared<LLMMLPNode>(OutputVector{input, gate_w, up_w, make_weights(kHidden, kUp, 0.3F)}, config);
        model_ref = std::make_shared<Model>(OutputVector{mlp}, ParameterVector{input});
    }
}

TEST_P(MLPFusionAfterActivationsScalingTest, Fused) {
    auto model = make_mlp(GetParam(), false);
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::ActivationsScaling>(kScale, element::f16);
    register_mlp_fusion(manager);
    manager.run_passes(model);

    size_t fused = 0;
    for (const auto& op : model->get_ops()) {
        fused += ov::is_type<LLMMLPNode>(op) ? 1 : 0;
    }
    EXPECT_EQ(fused, 1U);
}

const auto gate_up_types =
    testing::Values(GateUpType::SEPARATE, GateUpType::COMBINED_GATE_UP, GateUpType::COMBINED_UP_GATE);

INSTANTIATE_TEST_SUITE_P(TransformationTests, MLPFusionGateScaleTest, gate_up_types, type_name);
INSTANTIATE_TEST_SUITE_P(TransformationTests, MLPFusionAfterActivationsScalingTest, gate_up_types, type_name);
