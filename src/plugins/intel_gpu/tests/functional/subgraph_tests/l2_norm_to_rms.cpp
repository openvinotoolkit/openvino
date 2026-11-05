// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_sum.hpp"

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
}  // namespace
