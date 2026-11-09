// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/cpu_opset/x64/pass/mlp_fusion.hpp"

#include <gtest/gtest.h>

#include <memory>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/variadic_split.hpp"
#include "transformations/cpu_opset/x64/op/llm_mlp.hpp"

using namespace testing;
using namespace ov;
using namespace ov::op;

namespace {

constexpr size_t hidden_size = 64;
constexpr size_t up_size = 128;

struct GateUpMLPParams {
    PartialShape input_shape;
    element::Type weight_type;
    element::Type split_type;
    int64_t split_axis;
    std::vector<int64_t> split_lengths;
    size_t gate_up_rows = 2 * up_size;  // output features of the gate_up MatMul
    size_t down_cols = up_size;         // input features of the down MatMul
};

std::shared_ptr<v0::Constant> make_weight(element::Type type, const Shape& shape) {
    return v0::Constant::create(type, shape, std::vector<float>(shape_size(shape), 0.01f));
}

// gate_up MatMul -> VariadicSplit -> Swish(gate) * up -> down MatMul, weights decompressed to f32
std::shared_ptr<Model> make_gate_up_mlp(const GateUpMLPParams& p,
                                        std::shared_ptr<v0::Constant>& gate_up_w,
                                        std::shared_ptr<v0::Constant>& down_w,
                                        std::shared_ptr<v0::Parameter>& input) {
    input = std::make_shared<v0::Parameter>(element::f32, p.input_shape);
    gate_up_w = make_weight(p.weight_type, Shape{p.gate_up_rows, hidden_size});
    down_w = make_weight(p.weight_type, Shape{hidden_size, p.down_cols});

    auto gate_up =
        std::make_shared<v0::MatMul>(input, std::make_shared<v0::Convert>(gate_up_w, element::f32), false, true);
    auto split = std::make_shared<v1::VariadicSplit>(gate_up,
                                                     v0::Constant::create(p.split_type, Shape{}, {p.split_axis}),
                                                     v0::Constant::create(p.split_type, Shape{2}, p.split_lengths));
    auto gated = std::make_shared<v1::Multiply>(std::make_shared<v4::Swish>(split->output(0)), split->output(1));
    auto down = std::make_shared<v0::MatMul>(gated, std::make_shared<v0::Convert>(down_w, element::f32), false, true);
    return std::make_shared<Model>(OutputVector{down}, ParameterVector{input});
}

std::shared_ptr<Model> make_fused_mlp(const std::shared_ptr<v0::Parameter>& input,
                                      const std::shared_ptr<v0::Constant>& gate_up_w,
                                      const std::shared_ptr<v0::Constant>& down_w) {
    intel_cpu::LLMMLPNode::Config config{};
    config.act = intel_cpu::LLMMLPNode::ACT_FN::SILU;
    config.gate_up_quantized = false;
    config.down_quantized = false;
    config.hidden_size = static_cast<int>(hidden_size);
    config.up_size = static_cast<int>(up_size);
    config.gate_up_type = intel_cpu::LLMMLPNode::GATE_UP_TYPE::COMBINED_GATE_UP;
    auto mlp = std::make_shared<intel_cpu::LLMMLPNode>(OutputVector{input, gate_up_w, gate_up_w, down_w}, config);
    return std::make_shared<Model>(OutputVector{mlp}, ParameterVector{input});
}

void register_mlp_fusion(pass::Manager& manager) {
    manager.register_pass<intel_cpu::MLPFusion>();
    manager.get_pass_config()->set_callback<intel_cpu::MLPFusionPass>([](const std::shared_ptr<const Node>&) {
        return true;
    });
}

class MLPFusionGateUpTest : public TransformationTestsF, public WithParamInterface<GateUpMLPParams> {};

TEST_P(MLPFusionGateUpTest, Fused) {
    disable_rt_info_check();
    disable_result_friendly_names_check();
    std::shared_ptr<v0::Constant> gate_up_w, down_w;
    std::shared_ptr<v0::Parameter> input;
    model = make_gate_up_mlp(GetParam(), gate_up_w, down_w, input);
    register_mlp_fusion(manager);
    model_ref = make_fused_mlp(input, gate_up_w, down_w);
}

INSTANTIATE_TEST_SUITE_P(
    smoke,
    MLPFusionGateUpTest,
    Values(
        // Form produced by the common optimizations on a flattened [tokens, hidden] bf16 graph.
        GateUpMLPParams{PartialShape{-1, hidden_size}, element::bf16, element::i64, 1, {up_size, up_size}},
        // Pre-existing form: rank-3 input, f16 weights, i32 lengths, negative axis.
        GateUpMLPParams{PartialShape{-1, -1, hidden_size}, element::f16, element::i32, -1, {up_size, up_size}},
        GateUpMLPParams{PartialShape{-1, -1, hidden_size}, element::f16, element::i64, 2, {up_size, up_size}},
        // Inferred split length.
        GateUpMLPParams{PartialShape{-1, hidden_size}, element::bf16, element::i32, -1, {-1, up_size}}));

TEST_F(TransformationTestsF, MLPFusionGateUpUnequalSplitNotFused) {
    std::shared_ptr<v0::Constant> gate_up_w, down_w;
    std::shared_ptr<v0::Parameter> input;
    // [up, 1] split: the Multiply broadcasts, so the graph is valid, but it is not a gate/up pair.
    GateUpMLPParams p{PartialShape{-1, hidden_size}, element::bf16, element::i64, 1, {up_size, 1}};
    p.gate_up_rows = up_size + 1;
    model = make_gate_up_mlp(p, gate_up_w, down_w, input);
    register_mlp_fusion(manager);
}

TEST_F(TransformationTestsF, MLPFusionGateUpTokenAxisSplitNotFused) {
    std::shared_ptr<v0::Constant> gate_up_w, down_w;
    std::shared_ptr<v0::Parameter> input;
    // Splitting the token axis keeps all 2*up features in each half.
    GateUpMLPParams p{PartialShape{2 * up_size, hidden_size}, element::bf16, element::i64, 0, {up_size, up_size}};
    p.down_cols = 2 * up_size;
    model = make_gate_up_mlp(p, gate_up_w, down_w, input);
    register_mlp_fusion(manager);
}

}  // namespace
