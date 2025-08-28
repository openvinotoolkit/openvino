// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>

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