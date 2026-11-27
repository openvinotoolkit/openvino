// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "common_test_utils/test_common.hpp"
#include "transformations/cpu_opset/x64/pass/qkv_proj_fusion.hpp"
#include "transformations/cpu_opset/x64/op/qkv_proj.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/pass/visualize_tree.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "transformations/common_optimizations/activations_scaling.hpp"

using namespace testing;
using namespace ov::pass;
using namespace ov::op;
using namespace ov;

TEST_F(TransformationTestsF, QKVProjFusion1Test) {
    disable_rt_info_check();
    disable_result_friendly_names_check();

    auto is_quantized_int8 = false;
    size_t hidden_size = 2048;
    size_t q_proj_size = 2048;
    size_t k_proj_size = 256;
    size_t v_proj_size = 256;
    auto weights_combined = false;
    {
        auto input_multiply_const = std::make_shared<v0::Constant>(element::f32, Shape{1, 1, hidden_size});
        auto input_param = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, -1, static_cast<int>(hidden_size)});
        auto input_multiply = std::make_shared<v1::Multiply>(input_multiply_const, input_param);

        auto q_proj_weight_const = std::make_shared<v0::Constant>(element::f16, Shape{q_proj_size, hidden_size});
        auto q_proj_weight_cvt = std::make_shared<v0::Convert>(q_proj_weight_const, element::f32);
        auto q_proj = std::make_shared<v0::MatMul>(input_multiply, q_proj_weight_cvt, false, true);

        auto k_proj_weight_const = std::make_shared<v0::Constant>(element::f16, Shape{k_proj_size, hidden_size});
        auto k_proj_weight_cvt = std::make_shared<v0::Convert>(k_proj_weight_const, element::f32);
        auto k_proj = std::make_shared<v0::MatMul>(input_multiply, k_proj_weight_cvt, false, true);

        auto v_proj_weight_const = std::make_shared<v0::Constant>(element::f16, Shape{v_proj_size, hidden_size});
        auto v_proj_weight_cvt = std::make_shared<v0::Convert>(v_proj_weight_const, element::f32);
        auto v_proj = std::make_shared<v0::MatMul>(input_multiply, v_proj_weight_cvt, false, true);

        model = std::make_shared<ov::Model>(OutputVector{q_proj, k_proj, v_proj}, ParameterVector{input_param});
        manager.register_pass<ov::intel_cpu::QKVProjFusion>();
        manager.get_pass_config()->set_callback<ov::intel_cpu::QKVProjFusionPass1>(
            [=](const std::shared_ptr<const ov::Node>) -> bool {
                return true;
            });
    }
    {
        auto input_multiply_const = std::make_shared<v0::Constant>(element::f32, Shape{1, 1, hidden_size});
        auto input_param = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, -1, static_cast<int>(hidden_size)});
        auto input_multiply = std::make_shared<v1::Multiply>(input_multiply_const, input_param);

        auto q_proj_weight_const = std::make_shared<v0::Constant>(element::f16, Shape{q_proj_size, hidden_size});
        auto k_proj_weight_const = std::make_shared<v0::Constant>(element::f16, Shape{k_proj_size, hidden_size});
        auto v_proj_weight_const = std::make_shared<v0::Constant>(element::f16, Shape{v_proj_size, hidden_size});

        intel_cpu::QKVProjectionNode::Config config {is_quantized_int8, static_cast<int>(hidden_size),
                                                                        static_cast<int>(q_proj_size),
                                                                        static_cast<int>(k_proj_size),
                                                                        static_cast<int>(v_proj_size),
                                                                        weights_combined};
        auto qkv_proj = std::make_shared<intel_cpu::QKVProjectionNode>(OutputVector{input_multiply,
                                                                                    q_proj_weight_const,
                                                                                    k_proj_weight_const,
                                                                                    v_proj_weight_const},
                                                                                    config);

        auto q_proj = std::make_shared<v0::Result>(qkv_proj->output(0));
        auto k_proj = std::make_shared<v0::Result>(qkv_proj->output(1));
        auto v_proj = std::make_shared<v0::Result>(qkv_proj->output(2));
        model_ref = std::make_shared<ov::Model>(OutputVector{q_proj, k_proj, v_proj}, ParameterVector{input_param});
    }
}

namespace {
constexpr size_t kHidden = 64;
constexpr size_t kQProj = 64;
constexpr size_t kKVProj = 16;
constexpr float kScale = 8.0F;

std::shared_ptr<v0::Constant> make_proj_weights(size_t rows, float value) {
    return v0::Constant::create(element::f16, Shape{rows, kHidden}, std::vector<float>(rows * kHidden, value));
}
}  // namespace

// the Q/K/V MatMuls must still share one input after ActivationsScaling
TEST_F(TransformationTestsF, QKVProjFusionAfterActivationsScaling) {
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    {
        auto input = std::make_shared<v0::Parameter>(element::f16, PartialShape{-1, -1, kHidden});
        OutputVector outputs;
        for (const auto& [rows, value] : {std::pair{kQProj, 0.1F}, {kKVProj, 0.2F}, {kKVProj, 0.3F}}) {
            auto proj = std::make_shared<v0::MatMul>(input, make_proj_weights(rows, value), false, true);
            // the Convert keeps the output tensor name, as in the ActivationsScaling tests
            outputs.push_back(std::make_shared<v0::Convert>(proj, element::f32));
        }
        model = std::make_shared<ov::Model>(outputs, ParameterVector{input});
        manager.register_pass<ov::pass::ActivationsScaling>(kScale, element::f16);
        manager.register_pass<ov::intel_cpu::QKVProjFusion>();
        manager.get_pass_config()->set_callback<ov::intel_cpu::QKVProjFusionPass1>(
            [](const std::shared_ptr<const ov::Node>&) -> bool {
                return true;
            });
    }
    {
        // LPT keeps the scale Multiplies type-relaxed
        auto scale_mul = [](const Output<Node>& data, float value) {
            return std::make_shared<ov::op::TypeRelaxed<v1::Multiply>>(
                element::TypeVector{element::f16, element::f16},
                element::TypeVector{element::f16},
                data,
                v0::Constant::create(element::f16, Shape{}, {value}));
        };
        auto input = std::make_shared<v0::Parameter>(element::f16, PartialShape{-1, -1, kHidden});
        intel_cpu::QKVProjectionNode::Config config{false,
                                                    static_cast<int>(kHidden),
                                                    static_cast<int>(kQProj),
                                                    static_cast<int>(kKVProj),
                                                    static_cast<int>(kKVProj),
                                                    false};
        auto qkv_proj = std::make_shared<intel_cpu::QKVProjectionNode>(OutputVector{scale_mul(input, 1.0F / kScale),
                                                                                    make_proj_weights(kQProj, 0.1F),
                                                                                    make_proj_weights(kKVProj, 0.2F),
                                                                                    make_proj_weights(kKVProj, 0.3F)},
                                                                       config);
        OutputVector outputs;
        for (size_t i = 0; i < 3; ++i) {
            outputs.push_back(std::make_shared<v0::Convert>(scale_mul(qkv_proj->output(i), kScale), element::f32));
        }
        model_ref = std::make_shared<ov::Model>(outputs, ParameterVector{input});
    }
}
