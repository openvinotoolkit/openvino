// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/runtime/properties.hpp"

#include <memory>
#include <ostream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

namespace {
using ov::test::InputShape;

/*
 * The L2-normalization form of RMS norm produced by ONNX LpNormalization(p=2) exports:
 *
 *      Input(F32)      Const(F32)
 *           \            /
 *            Power(F32)  (x^2)
 *                |    Const(I64 axes=[-1])
 *                |       /
 *           ReduceSum(F32, keep_dims=true)
 *                |    Const(F32 eps)
 *                |      /
 *               Add(F32)
 *                |
 *              Sqrt(F32)
 *                |   \
 *   Input(F32) ->  Divide(F32)
 *                   |       \
 *                   |   Parameter gamma (F32)
 *                   |       /
 *                 Multiply(F32)
 *
 * ov::intel_gpu::L2NormToRMS rewrites it into a single RMS op
 * (RMS(x, gamma/sqrt(n), eps/n)). These tests exercise that rewrite end-to-end
 * on GPU against the CPU reference.
 */

enum class L2NormCase {
    // gamma is a proper last-axis scale: {N}
    Plain,
    // gamma product equals N but broadcasts along a non-last axis: {N,1}
    GammaColumn,
    // rank-6 activations, the rms primitive has no bfwzyx implementation
    RankSix,
};

using L2NormToRMSParams = std::tuple<L2NormCase,  // pattern variant
                                     std::vector<InputShape>,  // input shapes for {x, gamma}
                                     ov::element::Type>;       // input precision

class L2NormToRMS : public testing::WithParamInterface<L2NormToRMSParams>,
                    virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<L2NormToRMSParams>& obj) {
        const auto& [l2_case, input_shapes, input_precision] = obj.param;

        std::ostringstream result;
        switch (l2_case) {
        case L2NormCase::Plain: result << "Plain_"; break;
        case L2NormCase::GammaColumn: result << "GammaColumn_"; break;
        case L2NormCase::RankSix: result << "RankSix_"; break;
        }
        result << "IS=(";
        for (const auto& shape : input_shapes) {
            result << ov::test::utils::partialShape2str({shape.first}) << "_";
        }
        result << ")_TS=";
        for (const auto& shape : input_shapes) {
            result << "(";
            if (!shape.second.empty()) {
                auto itr = shape.second.begin();
                do {
                    result << ov::test::utils::vec2str(*itr);
                } while (++itr != shape.second.end() && result << "_");
            }
            result << ")_";
        }
        result << "input_precision=" << input_precision;
        return result.str();
    }

protected:
    L2NormCase l2_case = L2NormCase::Plain;

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;

        const auto& [l2case, input_shapes, input_precision] = GetParam();
        l2_case = l2case;

        init_input_shapes(input_shapes);

        inType = outType = input_precision;

        // tolerances are chosen so that the f16-output rounding of the fused RMS
        // (<= ~2e-4 here) stays green, while a broadcast-semantics mismatch (observed ~0.35) fails
        abs_threshold = 5e-3;
        rel_threshold = 1e-2;

        function = init_subgraph(inputDynamicShapes, input_precision);
    }

    std::shared_ptr<ov::Model> init_subgraph(const std::vector<ov::PartialShape>& input_shapes,
                                             const ov::element::Type& precision) {
        auto x = std::make_shared<ov::op::v0::Parameter>(precision, input_shapes[0]);
        auto gamma = std::make_shared<ov::op::v0::Parameter>(precision, input_shapes[1]);

        // x^2
        auto power_const = ov::op::v0::Constant::create(precision, {}, {2.f});
        auto power = std::make_shared<ov::op::v1::Power>(x, power_const);

        // ReduceSum(x^2, axes=[-1], keep_dims=true)
        auto axes_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
        auto sum = std::make_shared<ov::op::v1::ReduceSum>(power, axes_const, true);

        // Sqrt(ReduceSum + eps)
        auto eps = ov::op::v0::Constant::create(precision, {}, {1e-6f});
        auto add_eps = std::make_shared<ov::op::v1::Add>(sum, eps);
        auto sqrt = std::make_shared<ov::op::v0::Sqrt>(add_eps);

        // x / Sqrt(ReduceSum + eps) * gamma
        auto div = std::make_shared<ov::op::v1::Divide>(x, sqrt);
        auto mul = std::make_shared<ov::op::v1::Multiply>(div, gamma);

        auto model = std::make_shared<ov::Model>(ov::OutputVector{mul}, ov::ParameterVector{x, gamma}, "L2NormToRMS");
        model->get_parameters()[0]->set_friendly_name("x");
        model->get_parameters()[1]->set_friendly_name("gamma");
        return model;
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();

        auto x_tensor = ov::test::utils::create_and_fill_tensor(funcInputs[0].get_element_type(), targetInputStaticShapes[0]);
        inputs.insert({funcInputs[0].get_node_shared_ptr(), x_tensor});

        // Distinct per-index gamma values: for GammaColumn the row- and column-wise
        // broadcast interpretations cannot coincide (a uniform gamma would mask the bug).
        auto gamma_tensor = ov::Tensor(funcInputs[1].get_element_type(), targetInputStaticShapes[1]);
        const size_t count = gamma_tensor.get_size();
        const auto fill = [&](auto* data_ptr) {
            using T = std::decay_t<std::remove_pointer_t<decltype(data_ptr)>>;
            for (size_t i = 0; i < count; ++i) {
                data_ptr[i] = static_cast<T>(0.5f + 2.0f * static_cast<float>(i) / static_cast<float>(count));
            }
        };
        if (gamma_tensor.get_element_type() == ov::element::f16) {
            fill(gamma_tensor.data<ov::float16>());
        } else {
            fill(gamma_tensor.data<float>());
        }
        inputs.insert({funcInputs[1].get_node_shared_ptr(), gamma_tensor});
    }
};

TEST_P(L2NormToRMS, Inference) {
    run();
}

// Plain gamma {N} over {1,3,48}: the fused RMS must reproduce the decomposed form.
// x: 48 last dim -> n/8 = 6 <= max work group size, gamma size 48 == n -> the pass is expected to fire.
const std::vector<InputShape> input_shapes_plain = {
    {{1, 3, 48}, {{1, 3, 48}}},
    {{48}, {{48}}},
};

// gamma {48,1} over x {1,48,48}: same element count as the last dim, so the pass fires,
// but the original Multiply broadcasts gamma along axis 1 (per-row scale), while RMS
// consumes gamma as a flat vector along the normalized axis (per-column scale).
const std::vector<InputShape> input_shapes_gamma_column = {
    {{1, 48, 48}, {{1, 48, 48}}},
    {{48, 1}, {{48, 1}}},
};

// Rank-6 activations: rms has no bfwzyx implementation; the fused graph must still
// compile (either the pass declines the pattern or the plugin falls back), and must match CPU.
const std::vector<InputShape> input_shapes_rank_six = {
    {{1, 1, 1, 1, 4, 16}, {{1, 1, 1, 1, 4, 16}}},
    {{16}, {{16}}},
};

INSTANTIATE_TEST_SUITE_P(smoke_L2NormToRMS_plain,
                         L2NormToRMS,
                         ::testing::Combine(::testing::Values(L2NormCase::Plain),
                                            ::testing::Values(input_shapes_plain),
                                            ::testing::Values(ov::element::f32)),
                         L2NormToRMS::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_L2NormToRMS_gamma_column,
                         L2NormToRMS,
                         ::testing::Combine(::testing::Values(L2NormCase::GammaColumn),
                                            ::testing::Values(input_shapes_gamma_column),
                                            ::testing::Values(ov::element::f32)),
                         L2NormToRMS::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_L2NormToRMS_rank_six,
                         L2NormToRMS,
                         ::testing::Combine(::testing::Values(L2NormCase::RankSix),
                                            ::testing::Values(input_shapes_rank_six),
                                            ::testing::Values(ov::element::f32)),
                         L2NormToRMS::getTestCaseName);

// Finds the single rms layer in the runtime model. Comparing outputs alone also passes when the norm is not fused,
// so the tests check the fusion too.
void get_single_rms_layer(const ov::CompiledModel& compiled_model, std::shared_ptr<ov::Node>& rms) {
    const auto runtime_model = compiled_model.get_runtime_model();
    ASSERT_NE(runtime_model, nullptr);
    std::ostringstream layers;
    rms = nullptr;
    for (const auto& op : runtime_model->get_ordered_ops()) {
        const auto layer_type = op->get_rt_info().at("layerType").as<std::string>();
        layers << layer_type << " " << op->get_friendly_name() << " -> " << op->get_output_element_type(0) << "\n";
        if (layer_type == "rms") {
            ASSERT_EQ(rms, nullptr) << "expected a single rms layer";
            rms = op;
        }
    }
    ASSERT_NE(rms, nullptr) << "the L2 norm is not fused into RMS; runtime model:\n" << layers.str();
}

// Residual stream x0 + x1 + x2 + x3 feeding an L2 norm, f16 inference. With Overflow inputs the sum (~100000)
// exceeds f16, so the residual stream must stay FP32 (L2NormToRMS marks the fused RMS to stay in FP32).
enum class ResidualRange {
    InRange,   // inputs in [0, 1)
    Overflow,  // inputs in [20000, 30000)
};

// What lies between the residual Add and the norm.
enum class ResidualLink {
    Direct,
    BiasAdd,    // Add of a constant bias
    Reshape,    // rank-changing Reshape {1, 16, 64} -> {16, 64}
    Abs,        // unary elementwise arithmetic
    Broadcast,  // {1, 16, 64} -> {2, 16, 64}
};

const char* to_string(ResidualLink link) {
    switch (link) {
    case ResidualLink::Direct:
        return "Direct";
    case ResidualLink::BiasAdd:
        return "BiasAdd";
    case ResidualLink::Reshape:
        return "Reshape";
    case ResidualLink::Abs:
        return "Abs";
    case ResidualLink::Broadcast:
        return "Broadcast";
    }
    return "Unknown";
}

const std::vector<ResidualLink> residual_links{ResidualLink::Direct, ResidualLink::BiasAdd, ResidualLink::Reshape, ResidualLink::Abs, ResidualLink::Broadcast};

using ResidualOverflowParams = std::tuple<ResidualRange, ResidualLink>;

class L2NormToRMSResidualOverflow : public testing::WithParamInterface<ResidualOverflowParams>,
                                    virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ResidualOverflowParams>& obj) {
        const auto& [range, link] = obj.param;
        std::ostringstream name;
        name << (range == ResidualRange::Overflow ? "Overflow" : "InRange") << "_";
        name << to_string(link);
        return name.str();
    }

protected:
    static constexpr size_t num_inputs = 4;

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;
        configuration.insert({ov::hint::inference_precision.name(), ov::element::f16});
        // The InRange and Overflow cases compile the same model (also as the TEMPLATE reference). With a cache_dir left
        // on the shared core by an earlier test, the second compile imports the first's blob, whose inputs are other
        // Parameter nodes, and set_tensor fails. Restore the default (no model cache).
        core->set_property(ov::cache_dir(""));
        inType = outType = ov::element::f32;
        abs_threshold = 5e-3;
        rel_threshold = 1e-2;

        const ov::Shape shape{1, 16, 64};
        init_input_shapes(std::vector<InputShape>(num_inputs, {shape, {shape}}));

        // Each branch is a MatMul (identity weight, so the values stay those of the inputs and fit in f16): only the
        // residual Adds, not a model input, make the norm residual-fed.
        std::vector<float> identity(shape.back() * shape.back(), 0.f);
        for (size_t i = 0; i < shape.back(); ++i) {
            identity[i * shape.back() + i] = 1.f;
        }
        auto weight = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{shape.back(), shape.back()}, identity);
        ov::ParameterVector params;
        ov::OutputVector branches;
        for (size_t i = 0; i < num_inputs; ++i) {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape));
            branches.push_back(std::make_shared<ov::op::v0::MatMul>(params.back(), weight));
        }
        ov::Output<ov::Node> residual = branches[0];
        for (size_t i = 1; i < num_inputs; ++i) {
            residual = std::make_shared<ov::op::v1::Add>(residual, branches[i]);
        }
        const auto link = std::get<1>(GetParam());
        if (link == ResidualLink::BiasAdd) {
            residual = std::make_shared<ov::op::v1::Add>(residual, ov::op::v0::Constant::create(ov::element::f32, {shape.back()}, {0.25f}));
        } else if (link == ResidualLink::Reshape) {
            auto target_shape = ov::op::v0::Constant::create(ov::element::i64, {2}, {shape[1], shape[2]});
            residual = std::make_shared<ov::op::v1::Reshape>(residual, target_shape, false);
        } else if (link == ResidualLink::Abs) {
            residual = std::make_shared<ov::op::v0::Abs>(residual);
        } else if (link == ResidualLink::Broadcast) {
            auto target_shape = ov::op::v0::Constant::create(ov::element::i64, {3}, {size_t{2}, shape[1], shape[2]});
            residual = std::make_shared<ov::op::v3::Broadcast>(residual, target_shape);
        }

        // ONNX LpNormalization(p=2) lowering with FLT_EPSILON, as the ONNX frontend produces it
        auto power = std::make_shared<ov::op::v1::Power>(residual, ov::op::v0::Constant::create(ov::element::f32, {}, {2.f}));
        auto sum = std::make_shared<ov::op::v1::ReduceSum>(power, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}), true);
        auto add_eps = std::make_shared<ov::op::v1::Add>(sum, ov::op::v0::Constant::create(ov::element::f32, {}, {1.1920928955078125e-07f}));
        auto div = std::make_shared<ov::op::v1::Divide>(residual, std::make_shared<ov::op::v0::Sqrt>(add_eps));
        std::vector<float> gamma_values(shape.back());
        for (size_t i = 0; i < gamma_values.size(); ++i) {
            gamma_values[i] = 0.5f + 2.0f * static_cast<float>(i) / static_cast<float>(gamma_values.size());
        }
        auto gamma = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{shape.back()}, gamma_values);
        auto mul = std::make_shared<ov::op::v1::Multiply>(div, gamma);
        function = std::make_shared<ov::Model>(ov::OutputVector{mul}, params, "L2NormToRMSResidualOverflow");
    }

    void generate_inputs(const std::vector<ov::Shape>& target_input_static_shapes) override {
        inputs.clear();
        const auto& func_inputs = function->inputs();
        for (size_t i = 0; i < func_inputs.size(); ++i) {
            const auto seed = static_cast<int32_t>(i + 1);
            const auto gen = std::get<0>(GetParam()) == ResidualRange::Overflow ? ov::test::utils::InputGenerateData(20000, 10000, 1, seed)
                                                                   : ov::test::utils::InputGenerateData(0, 1, 1000, seed);
            inputs.insert({func_inputs[i].get_node_shared_ptr(),
                           ov::test::utils::create_and_fill_tensor(func_inputs[i].get_element_type(), target_input_static_shapes[i], gen)});
        }
    }

    // The norm is fused, and the RMS reads the residual stream in FP32.
    void validate() override {
        ov::test::SubgraphBaseTest::validate();
        std::shared_ptr<ov::Node> rms;
        ASSERT_NO_FATAL_FAILURE(get_single_rms_layer(compiledModel, rms));
        EXPECT_EQ(rms->get_input_element_type(0), ov::element::f32);
    }
};

TEST_P(L2NormToRMSResidualOverflow, Inference) {
    run();
}

INSTANTIATE_TEST_SUITE_P(smoke_L2NormToRMS_residual_overflow,
                         L2NormToRMSResidualOverflow,
                         ::testing::Combine(::testing::Values(ResidualRange::InRange, ResidualRange::Overflow), ::testing::ValuesIn(residual_links)),
                         L2NormToRMSResidualOverflow::getTestCaseName);

// L2 norm with gamma = (Convert(u16 weight) - zp) * scale, weight up to 65535, f16 inference. The u16 weight is a model
// input, as ONNX Runtime with the OpenVINO EP passes the gamma initializer, so KeepDequantizationPrecision (Constant
// weights only) does not keep it in FP32. The norm is fused before ConvertPrecision; a u16 -> f16 Convert of 65535
// gives inf, so the dequantization must stay FP32. x is a MatMul output, as for a PSX q/k norm: the RMS runs in f16,
// so only the gamma marks keep the dequantization in FP32.
class L2NormToRMSU16Gamma : public ov::test::SubgraphBaseStaticTest {
protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;
        configuration.insert({ov::hint::inference_precision.name(), ov::element::f16});
        outType = ov::element::f32;  // inputs keep their types: the gamma weight stays u16
        abs_threshold = 5e-3;
        rel_threshold = 1e-2;

        const ov::Shape shape{1, 16, 64};
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
        // Identity weight: the f16 MatMul is exact up to the f16 rounding of the input. With a mixed-sign weight, the
        // f16 MatMul error of near-zero outputs, scaled by the large gamma (up to ~56), exceeds the threshold on iGPU.
        std::vector<float> identity(shape.back() * shape.back(), 0.f);
        for (size_t i = 0; i < shape.back(); ++i) {
            identity[i * shape.back() + i] = 1.f;
        }
        auto x = std::make_shared<ov::op::v0::MatMul>(input, ov::op::v0::Constant::create(ov::element::f32, ov::Shape{shape.back(), shape.back()}, identity));

        auto weight = std::make_shared<ov::op::v0::Parameter>(ov::element::u16, ov::Shape{shape.back()});
        auto zero_point = ov::op::v0::Constant::create(ov::element::u16, ov::Shape{}, {127});
        auto shifted = std::make_shared<ov::op::v1::Subtract>(std::make_shared<ov::op::v0::Convert>(weight, ov::element::f32),
                                                              std::make_shared<ov::op::v0::Convert>(zero_point, ov::element::f32));
        auto gamma = std::make_shared<ov::op::v1::Multiply>(shifted, ov::op::v0::Constant::create(ov::element::f32, {}, {0.000855f}));

        // ONNX LpNormalization(p=2) lowering with FLT_EPSILON, as the ONNX frontend produces it
        auto power = std::make_shared<ov::op::v1::Power>(x, ov::op::v0::Constant::create(ov::element::f32, {}, {2.f}));
        auto sum = std::make_shared<ov::op::v1::ReduceSum>(power, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}), true);
        auto add_eps = std::make_shared<ov::op::v1::Add>(sum, ov::op::v0::Constant::create(ov::element::f32, {}, {1.1920928955078125e-07f}));
        auto div = std::make_shared<ov::op::v1::Divide>(x, std::make_shared<ov::op::v0::Sqrt>(add_eps));
        auto mul = std::make_shared<ov::op::v1::Multiply>(div, gamma);
        function = std::make_shared<ov::Model>(ov::OutputVector{mul}, ov::ParameterVector{input, weight}, "L2NormToRMSU16Gamma");
    }

    void generate_inputs(const std::vector<ov::Shape>& target_input_static_shapes) override {
        inputs.clear();
        const auto& func_inputs = function->inputs();
        inputs.insert({func_inputs[0].get_node_shared_ptr(),
                       ov::test::utils::create_and_fill_tensor(func_inputs[0].get_element_type(),
                                                               target_input_static_shapes[0],
                                                               ov::test::utils::InputGenerateData(0, 1, 1000, 1))});
        // Per-tensor u16 quantization of the gamma uses the full range: its largest element is 65535.
        ov::Tensor weight(ov::element::u16, target_input_static_shapes[1]);
        auto* weight_values = weight.data<uint16_t>();
        for (size_t i = 0; i < weight.get_size(); ++i) {
            weight_values[i] = static_cast<uint16_t>(20000 + 700 * i);
        }
        weight_values[weight.get_size() - 1] = 65535;
        inputs.insert({func_inputs[1].get_node_shared_ptr(), weight});
    }

    void validate() override {
        ov::test::SubgraphBaseStaticTest::validate();
        std::shared_ptr<ov::Node> rms;
        ASSERT_NO_FATAL_FAILURE(get_single_rms_layer(compiledModel, rms));
        EXPECT_EQ(rms->get_input_element_type(0), ov::element::f16);
    }
};

TEST_F(L2NormToRMSU16Gamma, smoke_Inference) {
    run();
}
// q/k-norm-like L2 norm fed by a MatMul (not a residual Add), f16 inference: the norm runs as RMS in f16. If it is
// left in FP32 (e.g. fused only after ConvertPrecision kept the L2-norm chain in FP32), the PSX latency gain is lost.
class L2NormToRMSNonResidualF16 : public ov::test::SubgraphBaseStaticTest {
protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;
        configuration.insert({ov::hint::inference_precision.name(), ov::element::f16});
        inType = outType = ov::element::f32;
        abs_threshold = 5e-3;
        rel_threshold = 1e-2;

        const ov::Shape shape{1, 16, 64};
        auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
        std::vector<float> weight_values(shape.back() * shape.back());
        for (size_t i = 0; i < weight_values.size(); ++i) {
            weight_values[i] = static_cast<float>(static_cast<int>(i % 13) - 6) / 64.f;
        }
        auto weight = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{shape.back(), shape.back()}, weight_values);
        auto q = std::make_shared<ov::op::v0::MatMul>(x, weight);

        // ONNX LpNormalization(p=2) lowering with FLT_EPSILON, as the ONNX frontend produces it
        auto power = std::make_shared<ov::op::v1::Power>(q, ov::op::v0::Constant::create(ov::element::f32, {}, {2.f}));
        auto sum = std::make_shared<ov::op::v1::ReduceSum>(power, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}), true);
        auto add_eps = std::make_shared<ov::op::v1::Add>(sum, ov::op::v0::Constant::create(ov::element::f32, {}, {1.1920928955078125e-07f}));
        auto div = std::make_shared<ov::op::v1::Divide>(q, std::make_shared<ov::op::v0::Sqrt>(add_eps));
        std::vector<float> gamma_values(shape.back());
        for (size_t i = 0; i < gamma_values.size(); ++i) {
            gamma_values[i] = 0.5f + 2.0f * static_cast<float>(i) / static_cast<float>(gamma_values.size());
        }
        auto gamma = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{shape.back()}, gamma_values);
        auto mul = std::make_shared<ov::op::v1::Multiply>(div, gamma);
        function = std::make_shared<ov::Model>(ov::OutputVector{mul}, ov::ParameterVector{x}, "L2NormToRMSNonResidualF16");
    }

    void validate() override {
        ov::test::SubgraphBaseStaticTest::validate();
        std::shared_ptr<ov::Node> rms;
        ASSERT_NO_FATAL_FAILURE(get_single_rms_layer(compiledModel, rms));
        EXPECT_EQ(rms->get_input_element_type(0), ov::element::f16);
        EXPECT_EQ(rms->get_output_element_type(0), ov::element::f16);
    }
};

TEST_F(L2NormToRMSNonResidualF16, smoke_Inference) {
    run();
}
}  // namespace
