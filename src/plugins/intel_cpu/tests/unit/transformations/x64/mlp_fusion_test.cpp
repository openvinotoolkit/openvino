// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/cpu_opset/x64/pass/mlp_fusion.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "common_test_utils/subgraph_builders/weights_decompression_builders.hpp"
#include "openvino/core/except.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/manager.hpp"
#include "ov_ops/type_relaxed.hpp"
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

struct MLPParams {
    GateUpType type;
    bool int8;
};

// rows in [scaled_begin, scaled_end) are scaled
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

// Convert(i8 [rows, kHidden]) * per-OC f32 scales [rows, 1]
std::shared_ptr<v1::Multiply> make_int8_weights(size_t rows, size_t seed) {
    using ov::test::utils::DecompressionType;
    auto deq = ov::test::utils::initMatMulDecompressionSubgraph(Shape{kHidden, rows},
                                                                -1,
                                                                element::f32,
                                                                element::i8,
                                                                element::f32,
                                                                element::f32,
                                                                true,
                                                                DecompressionType::full,
                                                                DecompressionType::empty,
                                                                false,
                                                                false,
                                                                seed);
    return ov::as_type_ptr<v1::Multiply>(deq);
}

// i8 weights and their per-OC scales, rows in [scaled_begin, scaled_end) scaled
std::pair<Output<Node>, Output<Node>> make_ref_int8_weights(size_t rows,
                                                            size_t seed,
                                                            size_t scaled_begin,
                                                            size_t scaled_end) {
    auto deq = make_int8_weights(rows, seed);
    auto scales = ov::as_type_ptr<v0::Constant>(deq->get_input_node_shared_ptr(1))->cast_vector<float>();
    for (size_t row = scaled_begin; row < scaled_end; ++row) {
        scales[row] *= kScale;
    }
    return {deq->get_input_node_ptr(0)->input_value(0), v0::Constant::create(element::f32, Shape{rows, 1}, scales)};
}

std::shared_ptr<Model> make_mlp(const MLPParams& params, bool scaled_gate) {
    const auto prc = params.int8 ? element::f32 : element::f16;
    auto weights = [&](size_t rows, float start, size_t seed) -> Output<Node> {
        if (params.int8) {
            return make_int8_weights(rows, seed);
        }
        return make_weights(rows, kHidden, start);
    };
    auto input = std::make_shared<v0::Parameter>(prc, PartialShape{-1, -1, kHidden});
    Output<Node> gate;
    Output<Node> up;
    if (params.type == GateUpType::SEPARATE) {
        gate = std::make_shared<v0::MatMul>(input, weights(kUp, 0.1F, 1), false, true);
        up = std::make_shared<v0::MatMul>(input, weights(kUp, 0.2F, 2), false, true);
    } else {
        auto gate_up = std::make_shared<v0::MatMul>(input, weights(2 * kUp, 0.1F, 1), false, true);
        auto split = std::make_shared<v1::VariadicSplit>(gate_up,
                                                         v0::Constant::create(element::i32, Shape{}, {-1}),
                                                         v0::Constant::create(element::i32, Shape{2}, {kUp, kUp}));
        const bool gate_first = params.type == GateUpType::COMBINED_GATE_UP;
        gate = split->output(gate_first ? 0 : 1);
        up = split->output(gate_first ? 1 : 0);
    }
    if (scaled_gate) {
        gate = std::make_shared<v1::Multiply>(gate, v0::Constant::create(prc, Shape{}, {kScale}));
    }
    auto gated = std::make_shared<v1::Multiply>(std::make_shared<v4::Swish>(gate), up);
    Output<Node> down_w = make_weights(kHidden, kUp, 0.3F);
    if (params.int8) {
        down_w = std::make_shared<v0::Convert>(down_w, element::f32);
    }
    auto down = std::make_shared<v0::MatMul>(gated, down_w, false, true);
    return std::make_shared<Model>(OutputVector{down}, ParameterVector{input});
}

// LLMMLP with the scale folded into the gate rows
std::shared_ptr<LLMMLPNode> make_ref_mlp(const Output<Node>& input, const MLPParams& params) {
    const size_t gate_begin = params.type == GateUpType::COMBINED_UP_GATE ? kUp : 0;
    const auto down_w = make_weights(kHidden, kUp, 0.3F);
    OutputVector args;
    if (params.int8 && params.type == GateUpType::SEPARATE) {
        auto gate = make_ref_int8_weights(kUp, 1, 0, kUp);
        auto up = make_ref_int8_weights(kUp, 2, 0, 0);
        args = {input, gate.first, up.first, down_w, gate.second, up.second};
    } else if (params.int8) {
        auto gate_up = make_ref_int8_weights(2 * kUp, 1, gate_begin, gate_begin + kUp);
        args = {input, gate_up.first, gate_up.first, down_w, gate_up.second, gate_up.second};
    } else if (params.type == GateUpType::SEPARATE) {
        args = {input, make_weights(kUp, kHidden, 0.1F, 0, kUp), make_weights(kUp, kHidden, 0.2F), down_w};
    } else {
        auto gate_up = make_weights(2 * kUp, kHidden, 0.1F, gate_begin, gate_begin + kUp);
        args = {input, gate_up, gate_up, down_w};
    }
    LLMMLPNode::Config config{};
    config.act = LLMMLPNode::ACT_FN::SILU;
    config.gate_up_quantized = params.int8;
    config.hidden_size = static_cast<int>(kHidden);
    config.up_size = static_cast<int>(kUp);
    config.gate_up_type = params.type;
    return std::make_shared<LLMMLPNode>(args, config);
}

void register_mlp_fusion(ov::pass::Manager& manager) {
    manager.register_pass<ov::intel_cpu::MLPFusion>();
    manager.get_pass_config()->set_callback<ov::intel_cpu::MLPFusionPass>([](const std::shared_ptr<const Node>&) {
        return true;
    });
}

std::string params_name(const testing::TestParamInfo<MLPParams>& info) {
    const std::string prc = info.param.int8 ? "Int8" : "F16";
    switch (info.param.type) {
    case GateUpType::SEPARATE:
        return prc + "Separate";
    case GateUpType::COMBINED_GATE_UP:
        return prc + "CombinedGateUp";
    case GateUpType::COMBINED_UP_GATE:
        return prc + "CombinedUpGate";
    default:
        OPENVINO_THROW("Unknown GateUpType");
    }
}

class MLPFusionGateScaleTest : public TransformationTestsF, public testing::WithParamInterface<MLPParams> {
protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    }
};

class MLPFusionAfterActivationsScalingTest : public MLPFusionGateScaleTest {};
}  // namespace

TEST_P(MLPFusionGateScaleTest, FoldedIntoGateRows) {
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    model = make_mlp(GetParam(), true);
    register_mlp_fusion(manager);
    {
        const auto prc = GetParam().int8 ? element::f32 : element::f16;
        auto input = std::make_shared<v0::Parameter>(prc, PartialShape{-1, -1, kHidden});
        model_ref = std::make_shared<Model>(OutputVector{make_ref_mlp(input, GetParam())}, ParameterVector{input});
    }
}

TEST_P(MLPFusionAfterActivationsScalingTest, Fused) {
    // the Convert keeps the output tensor name, as in the ActivationsScaling tests
    auto mlp = make_mlp(GetParam(), false);
    auto convert = std::make_shared<v0::Convert>(mlp->get_results()[0]->input_value(0), element::f32);
    model = std::make_shared<Model>(OutputVector{convert}, mlp->get_parameters());
    manager.register_pass<ov::pass::ActivationsScaling>(kScale, element::f16);
    register_mlp_fusion(manager);
    {
        auto input = std::make_shared<v0::Parameter>(element::f16, PartialShape{-1, -1, kHidden});
        // LPT keeps the scale Multiplies type-relaxed, whose attributes the comparator can't compare
        auto scale = [](const Output<Node>& data, float value) {
            return std::make_shared<ov::op::TypeRelaxed<v1::Multiply>>(
                element::TypeVector{element::f16, element::f16},
                element::TypeVector{element::f16},
                data,
                v0::Constant::create(element::f16, Shape{}, {value}));
        };
        auto scale_up = scale(make_ref_mlp(scale(input, 1.0F / kScale), GetParam()), kScale);
        auto convert = std::make_shared<v0::Convert>(scale_up, element::f32);
        model_ref = std::make_shared<Model>(OutputVector{convert}, ParameterVector{input});
    }
}

const std::vector<MLPParams> f16_params = {{GateUpType::SEPARATE, false},
                                           {GateUpType::COMBINED_GATE_UP, false},
                                           {GateUpType::COMBINED_UP_GATE, false}};

const std::vector<MLPParams> all_params = {{GateUpType::SEPARATE, false},
                                           {GateUpType::COMBINED_GATE_UP, false},
                                           {GateUpType::COMBINED_UP_GATE, false},
                                           {GateUpType::SEPARATE, true},
                                           {GateUpType::COMBINED_GATE_UP, true}};

INSTANTIATE_TEST_SUITE_P(TransformationTests, MLPFusionGateScaleTest, testing::ValuesIn(all_params), params_name);
// ActivationsScaling only scales f16
INSTANTIATE_TEST_SUITE_P(TransformationTests,
                         MLPFusionAfterActivationsScalingTest,
                         testing::ValuesIn(f16_params),
                         params_name);

// gate_up MatMul + VariadicSplit forms
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

class MLPFusionGateUpTest : public TransformationTestsF, public testing::WithParamInterface<GateUpMLPParams> {};

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
    testing::Values(
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
