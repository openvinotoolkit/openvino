// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "intel_gpu/op/moe_compressed.hpp"
#include "intel_gpu/op/moe_3gemm_fused_compressed.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/gather_elements.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "plugin/transformations/fuse_moe_3gemm_compressed.hpp"
#include "common_test_utils/node_builders/moe_builders.hpp"

using namespace testing;
using namespace ov::intel_gpu;

namespace ov {
namespace test {
namespace intel_gpu {

using namespace ov::test;

using FuseMOE3GemmCompressedTestParams = std::tuple<MoERoutingType, bool /* reshape_on_moe_input */>;

class FuseMOE3GemmCompressedTest : public TransformationTestsF,
                                   public ::testing::WithParamInterface<FuseMOE3GemmCompressedTestParams> {
public:
    static std::string get_test_case_name(const ::testing::TestParamInfo<FuseMOE3GemmCompressedTestParams>& info) {
        std::string name;
        switch (std::get<0>(info.param)) {
        case MoERoutingType::SOFTMAX:
            name = "Softmax";
            break;
        case MoERoutingType::SIGMOID_BIAS:
            name = "SigmoidBias";
            break;
        default:
            OPENVINO_THROW("Unsupported routing type");
        }
        if (std::get<1>(info.param))
            name += "_ReshapeOnMoeInput";
        return name;
    }
};

TEST_P(FuseMOE3GemmCompressedTest, CompareFunctions) {
    const auto& [routing_type, reshape_on_moe_input] = GetParam();
    {
        // tokens:32, hidden_size:2048, inter_size:768, experts:128, topk:8
        auto hidden_states = std::make_shared<ov::op::v0::Parameter>(element::f16, Shape{4, 8, 2048});
        auto flatten_shape = op::v0::Constant::create(element::i32, Shape{2}, {32, 2048});
        auto hidden_states_reshape = std::make_shared<ov::op::v1::Reshape>(hidden_states, flatten_shape, false);
        auto routers = op::v0::Constant::create(element::f16, Shape{2048, 128}, {0.2});
        auto routing_weights = std::make_shared<ov::op::v0::MatMul>(hidden_states_reshape, routers);

        // tokens:32, num_experts:128, topk:8
        auto routing_pair = routing_type == MoERoutingType::SOFTMAX
            ? build_softmax_routing_subgraph(routing_weights, 128, 8)
            : build_sigmoid_bias_routing_subgraph(routing_weights, element::f16, 128, 8);
        auto unsqueeze_moe = routing_pair.first;
        auto topk_indices = routing_pair.second;

        auto wei_gate = op::v0::Constant::create(element::u4, Shape{128, 768, 16, 128}, {1});
        auto scale_gate = op::v0::Constant::create(element::f16, Shape{128, 16, 768}, {0.01f});
        auto zp_gate = op::v0::Constant::create(element::u4, Shape{128, 16, 768}, {0});
        auto wei_up = op::v0::Constant::create(element::u4, Shape{128, 768, 16, 128}, {1});
        auto scale_up = op::v0::Constant::create(element::f16, Shape{128, 16, 768}, {0.01f});
        auto zp_up = op::v0::Constant::create(element::u4, Shape{128, 16, 768, 16}, {0});
        auto wei_down = op::v0::Constant::create(element::u4, Shape{128, 2048, 6, 128}, {1});
        auto scale_down = op::v0::Constant::create(element::f16, Shape{128, 6, 2048}, {0.01f});
        auto zp_down = op::v0::Constant::create(element::u4, Shape{128, 6, 2048}, {0});

        ov::intel_gpu::op::MOECompressed::Config config;
        config.hidden_size = 2048;
        config.inter_size = 768;
        config.num_expert = 128;
        config.group_size = 128;
        config.top_k = 8;
        config.out_type = ov::element::f16;

        // When reshape_on_moe_input is true, MOECompressed gets the flattened 2D hidden_states_reshape;
        // otherwise it gets the original 3D hidden_states.
        auto moe_input_0 = reshape_on_moe_input
            ? ov::Output<ov::Node>(hidden_states_reshape)
            : ov::Output<ov::Node>(hidden_states);
        auto moe_compressed = std::make_shared<ov::intel_gpu::op::MOECompressed>(
            ov::OutputVector{moe_input_0, unsqueeze_moe, topk_indices,
                wei_gate, scale_gate, zp_gate, wei_up, scale_up, zp_up, wei_down, scale_down, zp_down}, config);
        model = std::make_shared<ov::Model>(moe_compressed, ov::ParameterVector{hidden_states});
    }
    manager.register_pass<FuseMOE3GemmCompressed>();
    {
        // tokens:32, hidden_size:2048, inter_size:768, experts:128, topk:8
        auto hidden_states = std::make_shared<ov::op::v0::Parameter>(element::f16, Shape{4, 8, 2048});
        auto flatten_shape = op::v0::Constant::create(element::i32, Shape{2}, {32, 2048});
        auto hidden_states_reshape = std::make_shared<ov::op::v1::Reshape>(hidden_states, flatten_shape, false);
        auto routers = op::v0::Constant::create(element::f16, Shape{2048, 128}, {0.2});
        auto routing_weights = std::make_shared<ov::op::v0::MatMul>(hidden_states_reshape, routers);

        auto wei_gate = op::v0::Constant::create(element::u4, Shape{128, 768, 16, 128}, {1});
        auto scale_gate = op::v0::Constant::create(element::f16, Shape{128, 16, 768}, {0.01f});
        auto zp_gate = op::v0::Constant::create(element::u4, Shape{128, 16, 768}, {0});
        auto wei_up = op::v0::Constant::create(element::u4, Shape{128, 768, 16, 128}, {1});
        auto scale_up = op::v0::Constant::create(element::f16, Shape{128, 16, 768}, {0.01f});
        auto zp_up = op::v0::Constant::create(element::u4, Shape{128, 16, 768, 16}, {0});
        auto wei_down = op::v0::Constant::create(element::u4, Shape{128, 2048, 6, 128}, {1});
        auto scale_down = op::v0::Constant::create(element::f16, Shape{128, 6, 2048}, {0.01f});
        auto zp_down = op::v0::Constant::create(element::u4, Shape{128, 6, 2048}, {0});

        ov::intel_gpu::op::MOECompressed::Config config;
        config.hidden_size = 2048;
        config.inter_size = 768;
        config.num_expert = 128;
        config.group_size = 128;
        config.top_k = 8;
        config.out_type = ov::element::f16;
        if (routing_type == MoERoutingType::SOFTMAX) {
            config.routing_type = ov::intel_gpu::op::MOECompressed::RoutingType::SOFTMAX;
        } else if (routing_type == MoERoutingType::SIGMOID_BIAS) {
            config.routing_type = ov::intel_gpu::op::MOECompressed::RoutingType::SIGMOID_BIAS;
        } else {
            OPENVINO_THROW("Unsupported routing type");
        }

        ov::OutputVector args{hidden_states_reshape, routing_weights,
            wei_gate, scale_gate, zp_gate, wei_up, scale_up, zp_up, wei_down, scale_down, zp_down};
        if (routing_type == MoERoutingType::SIGMOID_BIAS) {
            auto routing_bias = op::v0::Constant::create(element::f16, Shape{1, 128}, {0.1f});
            args.push_back(routing_bias);
            auto routing_eps = op::v0::Constant::create(element::f16, Shape{1, 1}, {1e-6f});
            args.push_back(routing_eps);
        }

        std::shared_ptr<ov::Node> result = std::make_shared<ov::intel_gpu::op::MOE3GemmFusedCompressed>(args, config);

        // When reshape_on_moe_input is false, MOE3GemmFusedCompressed takes reshaped input from routing subgraph,
        // so the transformation inserts a reshape-back to restore the original shape.
        if (!reshape_on_moe_input) {
            auto hidden_state_shape = std::make_shared<ov::op::v3::ShapeOf>(hidden_states);
            result = std::make_shared<ov::op::v1::Reshape>(result, hidden_state_shape, false);
        }

        model_ref = std::make_shared<ov::Model>(result, ov::ParameterVector{hidden_states});
    }
}

INSTANTIATE_TEST_SUITE_P(smoke,
                         FuseMOE3GemmCompressedTest,
                         ::testing::Combine(
                             ::testing::Values(MoERoutingType::SOFTMAX, MoERoutingType::SIGMOID_BIAS),
                             ::testing::Values(false, true)),
                         FuseMOE3GemmCompressedTest::get_test_case_name);

TEST_F(TransformationTestsF, FuseMOE3GemmSharedExpertCompressedTest) {
    {
        // tokens:32, hidden_size:2048, iter_size:768, experts:128, topk:8
        auto hidden_states = std::make_shared<ov::op::v0::Parameter>(element::f16, Shape{32, 2048});
        auto routers = op::v0::Constant::create(element::f16, Shape{2048, 128}, {0.2});
        auto routing_weights = std::make_shared<ov::op::v0::MatMul>(hidden_states, routers);

        auto routing_pair = build_softmax_routing_subgraph(routing_weights, 128, 8);
        auto unsqueeze_moe = routing_pair.first;
        auto topk_indices = routing_pair.second;

        // weight
        auto wei_gate = op::v0::Constant::create(element::u4, Shape{128, 768, 16, 128}, {1});
        auto scale_gate = op::v0::Constant::create(element::f16, Shape{128, 16, 768}, {0.01f});
        auto zp_gate = op::v0::Constant::create(element::u4, Shape{128, 16, 768}, {0});
        auto wei_up = op::v0::Constant::create(element::u4, Shape{128, 768, 16, 128}, {1});
        auto scale_up = op::v0::Constant::create(element::f16, Shape{128, 16, 768}, {0.01f});
        auto zp_up = op::v0::Constant::create(element::u4, Shape{128, 16, 768, 16}, {0});
        auto wei_down = op::v0::Constant::create(element::u4, Shape{128, 2048, 6, 128}, {1});
        auto scale_down = op::v0::Constant::create(element::f16, Shape{128, 6, 2048}, {0.01f});
        auto zp_down = op::v0::Constant::create(element::u4, Shape{128, 6, 2048}, {0});

        // Shared expert weights (single shared expert: leading dimension 1)
        auto sh_wei_gate = op::v0::Constant::create(element::u4, Shape{1, 768, 16, 128}, {2});
        auto sh_scale_gate = op::v0::Constant::create(element::f16, Shape{1, 16, 768}, {0.02f});
        auto sh_zp_gate = op::v0::Constant::create(element::u4, Shape{1, 16, 768}, {0});
        auto sh_wei_up = op::v0::Constant::create(element::u4, Shape{1, 768, 16, 128}, {2});
        auto sh_scale_up = op::v0::Constant::create(element::f16, Shape{1, 16, 768}, {0.02f});
        auto sh_zp_up = op::v0::Constant::create(element::u4, Shape{1, 16, 768, 16}, {0});
        auto sh_wei_down = op::v0::Constant::create(element::u4, Shape{1, 2048, 6, 128}, {2});
        auto sh_scale_down = op::v0::Constant::create(element::f16, Shape{1, 6, 2048}, {0.02f});
        auto sh_zp_down = op::v0::Constant::create(element::u4, Shape{1, 6, 2048}, {0});
        auto sh_gate_gate_wei = op::v0::Constant::create(element::f16, Shape{2048, 1}, {0.5f});

        ov::intel_gpu::op::MOECompressed::Config config;
        config.hidden_size = 2048;
        config.inter_size = 768;
        config.num_expert = 128;
        config.num_shared_expert = 1;
        config.group_size = 128;
        config.top_k = 8;
        config.out_type = ov::element::f16;
        auto moe_compressed = std::make_shared<ov::intel_gpu::op::MOECompressed>(
            ov::OutputVector{hidden_states, unsqueeze_moe, topk_indices,
                wei_gate, scale_gate, zp_gate, wei_up, scale_up, zp_up, wei_down, scale_down, zp_down,
                sh_wei_gate, sh_scale_gate, sh_zp_gate, sh_wei_up, sh_scale_up, sh_zp_up,
                sh_wei_down, sh_scale_down, sh_zp_down, sh_gate_gate_wei}, config);
        model = std::make_shared<ov::Model>(moe_compressed, ov::ParameterVector{hidden_states});
        manager.register_pass<FuseMOE3GemmCompressed>();
    }
    {
        // tokens:32, hidden_size:2048, iter_size:768, experts:128, topk:8
        auto hidden_states = std::make_shared<ov::op::v0::Parameter>(element::f16, Shape{32, 2048});
        auto routers = op::v0::Constant::create(element::f16, Shape{2048, 128}, {0.2});
        auto routing_weights = std::make_shared<ov::op::v0::MatMul>(hidden_states, routers);

        // MOE expert weights
        auto wei_gate = op::v0::Constant::create(element::u4, Shape{128, 768, 16, 128}, {1});
        auto scale_gate = op::v0::Constant::create(element::f16, Shape{128, 16, 768}, {0.01f});
        auto zp_gate = op::v0::Constant::create(element::u4, Shape{128, 16, 768}, {0});
        auto wei_up = op::v0::Constant::create(element::u4, Shape{128, 768, 16, 128}, {1});
        auto scale_up = op::v0::Constant::create(element::f16, Shape{128, 16, 768}, {0.01f});
        auto zp_up = op::v0::Constant::create(element::u4, Shape{128, 16, 768, 16}, {0});
        auto wei_down = op::v0::Constant::create(element::u4, Shape{128, 2048, 6, 128}, {1});
        auto scale_down = op::v0::Constant::create(element::f16, Shape{128, 6, 2048}, {0.01f});
        auto zp_down = op::v0::Constant::create(element::u4, Shape{128, 6, 2048}, {0});

        // Shared expert weights (single shared expert: leading dimension 1)
        auto sh_wei_gate = op::v0::Constant::create(element::u4, Shape{1, 768, 16, 128}, {2});
        auto sh_scale_gate = op::v0::Constant::create(element::f16, Shape{1, 16, 768}, {0.02f});
        auto sh_zp_gate = op::v0::Constant::create(element::u4, Shape{1, 16, 768}, {0});
        auto sh_wei_up = op::v0::Constant::create(element::u4, Shape{1, 768, 16, 128}, {2});
        auto sh_scale_up = op::v0::Constant::create(element::f16, Shape{1, 16, 768}, {0.02f});
        auto sh_zp_up = op::v0::Constant::create(element::u4, Shape{1, 16, 768, 16}, {0});
        auto sh_wei_down = op::v0::Constant::create(element::u4, Shape{1, 2048, 6, 128}, {2});
        auto sh_scale_down = op::v0::Constant::create(element::f16, Shape{1, 6, 2048}, {0.02f});
        auto sh_zp_down = op::v0::Constant::create(element::u4, Shape{1, 6, 2048}, {0});
        auto sh_gate_gate_wei = op::v0::Constant::create(element::f16, Shape{2048, 1}, {0.5f});

        // Dummy placeholders for SOFTMAX + shared expert (slots 11, 12, 13)
        auto dummy_bias       = op::v0::Constant::create(element::f16, Shape{1}, {0.0f});
        auto dummy_eps        = op::v0::Constant::create(element::f16, Shape{1}, {0.0f});
        auto dummy_norm_scale = op::v0::Constant::create(element::f16, Shape{1}, {1.0f});

        ov::intel_gpu::op::MOECompressed::Config config;
        config.hidden_size = 2048;
        config.inter_size = 768;
        config.num_expert = 128;
        config.num_shared_expert = 1;
        config.group_size = 128;
        config.top_k = 8;
        config.out_type = ov::element::f16;
        // 24 inputs: [0-10] expert, [11] dummy_bias, [12] dummy_eps, [13] dummy_norm_scale, [14-23] shared
        auto moe_3gemm_fused_compressed = std::make_shared<ov::intel_gpu::op::MOE3GemmFusedCompressed>(
            ov::OutputVector{hidden_states, routing_weights,
                wei_gate, scale_gate, zp_gate, wei_up, scale_up, zp_up, wei_down, scale_down, zp_down,
                dummy_bias, dummy_eps, dummy_norm_scale,
                sh_wei_gate, sh_scale_gate, sh_zp_gate, sh_wei_up, sh_scale_up, sh_zp_up,
                sh_wei_down, sh_scale_down, sh_zp_down, sh_gate_gate_wei}, config);

        model_ref = std::make_shared<ov::Model>(moe_3gemm_fused_compressed, ov::ParameterVector{hidden_states});
    }
}

TEST_F(TransformationTestsF, FuseMOE3GemmCompressed_SigmoidBias_ScaledNorm_SharedExpert) {
    // Verifies that FuseMOE3GemmCompressed fires when both:
    //   (a) sigmoid routing has a post-normalization Multiply(Divide, Constant) (has_routing_norm_scale), AND
    //   (b) the MOECompressed input includes shared expert weights (num_shared_expert=1).
    // Expected output: MOE3GemmFusedCompressed with 24 inputs, routing_type=SIGMOID_BIAS,
    //   has_routing_norm_scale=true, num_shared_expert=1.
    //   Input layout: [0-10] expert weights, [11] bias, [12] eps, [13] norm_scale, [14-23] shared expert.
    {
        auto hidden_states = std::make_shared<ov::op::v0::Parameter>(element::f16, Shape{32, 2048});
        auto routers = op::v0::Constant::create(element::f16, Shape{2048, 128}, {0.2f});
        auto routing_weights = std::make_shared<ov::op::v0::MatMul>(hidden_states, routers);

        auto routing_pair = build_sigmoid_bias_scaled_norm_routing_subgraph(routing_weights, element::f16, 128, 8);
        auto unsqueeze_moe = routing_pair.first;
        auto topk_indices = routing_pair.second;

        // Expert weights
        auto wei_gate = op::v0::Constant::create(element::u4, Shape{128, 768, 16, 128}, {1});
        auto scale_gate = op::v0::Constant::create(element::f16, Shape{128, 16, 768}, {0.01f});
        auto zp_gate = op::v0::Constant::create(element::u4, Shape{128, 16, 768}, {0});
        auto wei_up = op::v0::Constant::create(element::u4, Shape{128, 768, 16, 128}, {1});
        auto scale_up = op::v0::Constant::create(element::f16, Shape{128, 16, 768}, {0.01f});
        auto zp_up = op::v0::Constant::create(element::u4, Shape{128, 16, 768, 16}, {0});
        auto wei_down = op::v0::Constant::create(element::u4, Shape{128, 2048, 6, 128}, {1});
        auto scale_down = op::v0::Constant::create(element::f16, Shape{128, 6, 2048}, {0.01f});
        auto zp_down = op::v0::Constant::create(element::u4, Shape{128, 6, 2048}, {0});

        // Shared expert weights
        auto sh_wei_gate = op::v0::Constant::create(element::u4, Shape{1, 768, 16, 128}, {2});
        auto sh_scale_gate = op::v0::Constant::create(element::f16, Shape{1, 16, 768}, {0.02f});
        auto sh_zp_gate = op::v0::Constant::create(element::u4, Shape{1, 16, 768}, {0});
        auto sh_wei_up = op::v0::Constant::create(element::u4, Shape{1, 768, 16, 128}, {2});
        auto sh_scale_up = op::v0::Constant::create(element::f16, Shape{1, 16, 768}, {0.02f});
        auto sh_zp_up = op::v0::Constant::create(element::u4, Shape{1, 16, 768, 16}, {0});
        auto sh_wei_down = op::v0::Constant::create(element::u4, Shape{1, 2048, 6, 128}, {2});
        auto sh_scale_down = op::v0::Constant::create(element::f16, Shape{1, 6, 2048}, {0.02f});
        auto sh_zp_down = op::v0::Constant::create(element::u4, Shape{1, 6, 2048}, {0});
        auto sh_gate_gate_wei = op::v0::Constant::create(element::f16, Shape{2048, 1}, {0.5f});

        ov::intel_gpu::op::MOECompressed::Config config;
        config.hidden_size = 2048;
        config.inter_size = 768;
        config.num_expert = 128;
        config.num_shared_expert = 1;
        config.group_size = 128;
        config.top_k = 8;
        config.out_type = ov::element::f16;

        auto moe_compressed = std::make_shared<ov::intel_gpu::op::MOECompressed>(
            ov::OutputVector{hidden_states, unsqueeze_moe, topk_indices,
                wei_gate, scale_gate, zp_gate, wei_up, scale_up, zp_up, wei_down, scale_down, zp_down,
                sh_wei_gate, sh_scale_gate, sh_zp_gate, sh_wei_up, sh_scale_up, sh_zp_up,
                sh_wei_down, sh_scale_down, sh_zp_down, sh_gate_gate_wei}, config);
        model = std::make_shared<ov::Model>(moe_compressed, ov::ParameterVector{hidden_states});
        manager.register_pass<FuseMOE3GemmCompressed>();
    }
    {
        // Expected: 24-input MOE3GemmFusedCompressed.
        // Layout: [0] hidden, [1] routing_weights(MatMul), [2-10] expert w/s/zp,
        //         [11] bias, [12] eps, [13] norm_scale, [14-23] shared expert.
        auto hidden_states = std::make_shared<ov::op::v0::Parameter>(element::f16, Shape{32, 2048});
        auto routers = op::v0::Constant::create(element::f16, Shape{2048, 128}, {0.2f});
        auto routing_weights_ref = std::make_shared<ov::op::v0::MatMul>(hidden_states, routers);

        auto wei_gate = op::v0::Constant::create(element::u4, Shape{128, 768, 16, 128}, {1});
        auto scale_gate = op::v0::Constant::create(element::f16, Shape{128, 16, 768}, {0.01f});
        auto zp_gate = op::v0::Constant::create(element::u4, Shape{128, 16, 768}, {0});
        auto wei_up = op::v0::Constant::create(element::u4, Shape{128, 768, 16, 128}, {1});
        auto scale_up = op::v0::Constant::create(element::f16, Shape{128, 16, 768}, {0.01f});
        auto zp_up = op::v0::Constant::create(element::u4, Shape{128, 16, 768, 16}, {0});
        auto wei_down = op::v0::Constant::create(element::u4, Shape{128, 2048, 6, 128}, {1});
        auto scale_down = op::v0::Constant::create(element::f16, Shape{128, 6, 2048}, {0.01f});
        auto zp_down = op::v0::Constant::create(element::u4, Shape{128, 6, 2048}, {0});

        auto sh_wei_gate = op::v0::Constant::create(element::u4, Shape{1, 768, 16, 128}, {2});
        auto sh_scale_gate = op::v0::Constant::create(element::f16, Shape{1, 16, 768}, {0.02f});
        auto sh_zp_gate = op::v0::Constant::create(element::u4, Shape{1, 16, 768}, {0});
        auto sh_wei_up = op::v0::Constant::create(element::u4, Shape{1, 768, 16, 128}, {2});
        auto sh_scale_up = op::v0::Constant::create(element::f16, Shape{1, 16, 768}, {0.02f});
        auto sh_zp_up = op::v0::Constant::create(element::u4, Shape{1, 16, 768, 16}, {0});
        auto sh_wei_down = op::v0::Constant::create(element::u4, Shape{1, 2048, 6, 128}, {2});
        auto sh_scale_down = op::v0::Constant::create(element::f16, Shape{1, 6, 2048}, {0.02f});
        auto sh_zp_down = op::v0::Constant::create(element::u4, Shape{1, 6, 2048}, {0});
        auto sh_gate_gate_wei = op::v0::Constant::create(element::f16, Shape{2048, 1}, {0.5f});

        // Routing params captured from the scaled-norm sigmoid subgraph
        auto ref_bias       = op::v0::Constant::create(element::f16, Shape{1, 128}, {0.0f});   // sig_routing_bias
        auto ref_eps        = op::v0::Constant::create(element::f16, Shape{1, 1},   {1e-6f});   // sig_eps_value
        auto ref_norm_scale = op::v0::Constant::create(element::f16, Shape{1},      {1.0f});    // sig_norm_scale_const

        ov::intel_gpu::op::MOECompressed::Config config;
        config.hidden_size            = 2048;
        config.inter_size             = 768;
        config.num_expert             = 128;
        config.num_shared_expert      = 1;
        config.group_size             = 128;
        config.top_k                  = 8;
        config.out_type               = ov::element::f16;
        config.routing_type           = ov::intel_gpu::op::MOECompressed::RoutingType::SIGMOID_BIAS;
        config.has_routing_norm_scale = true;

        // 24 inputs: [0-10] expert, [11] bias, [12] eps, [13] norm_scale, [14-23] shared
        auto moe_3gemm_fused_compressed = std::make_shared<ov::intel_gpu::op::MOE3GemmFusedCompressed>(
            ov::OutputVector{hidden_states, routing_weights_ref,
                wei_gate, scale_gate, zp_gate, wei_up, scale_up, zp_up, wei_down, scale_down, zp_down,
                ref_bias, ref_eps, ref_norm_scale,
                sh_wei_gate, sh_scale_gate, sh_zp_gate, sh_wei_up, sh_scale_up, sh_zp_up,
                sh_wei_down, sh_scale_down, sh_zp_down, sh_gate_gate_wei}, config);

        model_ref = std::make_shared<ov::Model>(moe_3gemm_fused_compressed, ov::ParameterVector{hidden_states});
    }
}

TEST_F(TransformationTestsF, FuseMOE3GemmCompressed_SigmoidBias_ScaledNorm) {
    // Reproducer: sigmoid routing branch has Multiply(Divide, Constant) inserted between
    // the normalization Divide and the Slice.  FuseMOE3GemmCompressed pattern expects
    // Slice(Divide(...)) but the graph has Slice(Multiply(Divide(...), Constant)).
    //
    // Root-cause log phrase:
    //   "NODES' TYPE DIDN'T MATCH. EXPECTED: WrapType<Divide>. OBSERVED: Multiply"
    //   (at ARGUMENT 0 of sig_slice, which is ARGUMENT 2 of ScatterElementsUpdate)
    //
    // Before fix: test passes GREEN (transformation does not fire, model is unchanged).
    // After fix:  test FAILS because model != auto-cloned ref → add explicit model_ref.
    {
        auto hidden_states =
            std::make_shared<ov::op::v0::Parameter>(element::f16, Shape{32, 2048});
        auto routers = op::v0::Constant::create(element::f16, Shape{2048, 128}, {0.2f});
        auto routing_matmul = std::make_shared<ov::op::v0::MatMul>(hidden_states, routers);

        // Sigmoid routing with bias
        auto sigmoid = std::make_shared<ov::op::v0::Sigmoid>(routing_matmul);
        auto bias = op::v0::Constant::create(element::f16, Shape{1, 128}, {0.1f});
        auto sig_add = std::make_shared<ov::op::v1::Add>(sigmoid, bias);
        auto k_const = op::v0::Constant::create(element::i64, Shape{}, {8});
        auto topk = std::make_shared<ov::op::v11::TopK>(
                sig_add, k_const, -1,
                ov::op::v11::TopK::Mode::MAX, ov::op::v11::TopK::SortType::SORT_VALUES,
                element::i64);
        auto convert_topk = std::make_shared<ov::op::v0::Convert>(topk->output(1), element::i32);

        // Normalization: GatherElements -> ReduceSum -> Add(eps) -> Divide
        auto gather_el = std::make_shared<ov::op::v6::GatherElements>(sigmoid, convert_topk, 1);
        auto reduce = std::make_shared<ov::op::v1::ReduceSum>(
                gather_el,
                op::v0::Constant::create(element::i64, Shape{1}, {1}),
                true);
        auto eps = op::v0::Constant::create(element::f16, Shape{1, 1}, {1e-6f});
        auto add_eps = std::make_shared<ov::op::v1::Add>(reduce, eps);
        auto norm = std::make_shared<ov::op::v1::Divide>(gather_el, add_eps);

        // *** Extra Multiply(Divide, Constant) — the root cause of the pattern mismatch ***
        auto scale = op::v0::Constant::create(element::f16, Shape{1}, {1.0f});
        auto norm_scaled = std::make_shared<ov::op::v1::Multiply>(norm, scale);

        // Slice now feeds from norm_scaled (Multiply) instead of norm (Divide)
        auto sl_stop = std::make_shared<ov::op::v3::ShapeOf>(convert_topk, element::i32);
        auto scatter_w = std::make_shared<ov::op::v8::Slice>(
                norm_scaled,
                op::v0::Constant::create(element::i32, Shape{2}, {0, 0}),
                sl_stop,
                op::v0::Constant::create(element::i32, Shape{2}, {1, 1}),
                op::v0::Constant::create(element::i64, Shape{2}, {0, 1}));

        // Build scatter + transpose + reshape + unsqueeze (MOECompressed input[1])
        auto ne_c = op::v0::Constant::create(element::i64, Shape{}, {128});
        auto unsq_ne = std::make_shared<ov::op::v0::Unsqueeze>(
                ne_c, op::v0::Constant::create(element::i64, Shape{1}, {0}));
        auto shapeof = std::make_shared<ov::op::v3::ShapeOf>(convert_topk, element::i64);
        auto gather_s = std::make_shared<ov::op::v8::Gather>(
                shapeof,
                op::v0::Constant::create(element::i64, Shape{}, {0}),
                op::v0::Constant::create(element::i64, Shape{}, {0}));
        auto unsq_seq = std::make_shared<ov::op::v0::Unsqueeze>(
                gather_s, op::v0::Constant::create(element::i64, Shape{1}, {0}));
        auto bcast_shape =
                std::make_shared<ov::op::v0::Concat>(ov::OutputVector{unsq_seq, unsq_ne}, 0);
        auto zero = op::v0::Constant::create(element::f16, Shape{1}, {0.0f});
        auto bcast = std::make_shared<ov::op::v3::Broadcast>(zero, bcast_shape);
        auto scatter = std::make_shared<ov::op::v12::ScatterElementsUpdate>(
                bcast, convert_topk, scatter_w,
                op::v0::Constant::create(element::i64, Shape{}, {1}));
        auto transp = std::make_shared<ov::op::v1::Transpose>(
                scatter, op::v0::Constant::create(element::i64, Shape{2}, {1, 0}));
        auto reshape_c = op::v0::Constant::create(element::i64, Shape{3}, {128, 32, 1});
        auto reshape = std::make_shared<ov::op::v1::Reshape>(transp, reshape_c, false);
        auto unsqueeze_moe = std::make_shared<ov::op::v0::Unsqueeze>(
                reshape, op::v0::Constant::create(element::i64, Shape{1}, {3}));

        auto wei_gate   = op::v0::Constant::create(element::u4,  Shape{128, 768, 16, 128}, {1});
        auto scale_gate = op::v0::Constant::create(element::f16, Shape{128, 16, 768},       {0.01f});
        auto zp_gate    = op::v0::Constant::create(element::u4,  Shape{128, 16, 768},        {0});
        auto wei_up     = op::v0::Constant::create(element::u4,  Shape{128, 768, 16, 128},  {1});
        auto scale_up   = op::v0::Constant::create(element::f16, Shape{128, 16, 768},        {0.01f});
        auto zp_up      = op::v0::Constant::create(element::u4,  Shape{128, 16, 768, 16},   {0});
        auto wei_down   = op::v0::Constant::create(element::u4,  Shape{128, 2048, 6, 128},  {1});
        auto scale_down = op::v0::Constant::create(element::f16, Shape{128, 6, 2048},        {0.01f});
        auto zp_down    = op::v0::Constant::create(element::u4,  Shape{128, 6, 2048},        {0});

        ov::intel_gpu::op::MOECompressed::Config config;
        config.hidden_size  = 2048;
        config.inter_size   = 768;
        config.num_expert   = 128;
        config.group_size   = 128;
        config.top_k        = 8;
        config.out_type     = ov::element::f16;
        config.routing_type = ov::intel_gpu::op::MOECompressed::RoutingType::SIGMOID_BIAS;

        auto moe_compressed = std::make_shared<ov::intel_gpu::op::MOECompressed>(
                ov::OutputVector{hidden_states, unsqueeze_moe, convert_topk,
                    wei_gate, scale_gate, zp_gate, wei_up, scale_up, zp_up,
                    wei_down, scale_down, zp_down},
                config);
        model = std::make_shared<ov::Model>(moe_compressed, ov::ParameterVector{hidden_states});
        manager.register_pass<FuseMOE3GemmCompressed>();
    }
    {
        // Expected output after transformation: MOE3GemmFusedCompressed with 14 inputs.
        // Input 13 is the routing_norm_scale constant captured from the Multiply node.
        auto hidden_states  = std::make_shared<ov::op::v0::Parameter>(element::f16, Shape{32, 2048});
        auto routers        = op::v0::Constant::create(element::f16, Shape{2048, 128}, {0.2f});
        auto routing_matmul = std::make_shared<ov::op::v0::MatMul>(hidden_states, routers);

        auto wei_gate   = op::v0::Constant::create(element::u4,  Shape{128, 768, 16, 128}, {1});
        auto scale_gate = op::v0::Constant::create(element::f16, Shape{128, 16, 768},       {0.01f});
        auto zp_gate    = op::v0::Constant::create(element::u4,  Shape{128, 16, 768},        {0});
        auto wei_up     = op::v0::Constant::create(element::u4,  Shape{128, 768, 16, 128},  {1});
        auto scale_up   = op::v0::Constant::create(element::f16, Shape{128, 16, 768},        {0.01f});
        auto zp_up      = op::v0::Constant::create(element::u4,  Shape{128, 16, 768, 16},   {0});
        auto wei_down   = op::v0::Constant::create(element::u4,  Shape{128, 2048, 6, 128},  {1});
        auto scale_down = op::v0::Constant::create(element::f16, Shape{128, 6, 2048},        {0.01f});
        auto zp_down    = op::v0::Constant::create(element::u4,  Shape{128, 6, 2048},        {0});

        // Routing parameters captured from the sigmoid branch
        auto ref_bias   = op::v0::Constant::create(element::f16, Shape{1, 128},  {0.1f});   // sig_routing_bias
        auto ref_eps    = op::v0::Constant::create(element::f16, Shape{1, 1},    {1e-6f});   // sig_eps_value
        auto ref_scale  = op::v0::Constant::create(element::f16, Shape{1},       {1.0f});    // sig_norm_scale_const

        ov::intel_gpu::op::MOECompressed::Config config;
        config.hidden_size           = 2048;
        config.inter_size            = 768;
        config.num_expert            = 128;
        config.group_size            = 128;
        config.top_k                 = 8;
        config.out_type              = ov::element::f16;
        config.routing_type          = ov::intel_gpu::op::MOECompressed::RoutingType::SIGMOID_BIAS;
        config.has_routing_norm_scale = true;

        auto moe_3gemm = std::make_shared<ov::intel_gpu::op::MOE3GemmFusedCompressed>(
                ov::OutputVector{hidden_states, routing_matmul,
                    wei_gate, scale_gate, zp_gate, wei_up, scale_up, zp_up,
                    wei_down, scale_down, zp_down,
                    ref_bias, ref_eps, ref_scale},
                config);
        model_ref = std::make_shared<ov::Model>(moe_3gemm, ov::ParameterVector{hidden_states});
    }
}

}  // namespace intel_gpu
}  // namespace test
}  // namespace ov
