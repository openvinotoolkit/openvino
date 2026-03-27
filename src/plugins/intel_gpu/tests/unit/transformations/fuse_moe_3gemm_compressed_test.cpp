// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "intel_gpu/op/moe_compressed.hpp"
#include "intel_gpu/op/moe_3gemm_fused_compressed.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather_elements.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/softmax.hpp"
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

/// Build the post-GatherMatmul softmax routing chain that FuseMOE3GemmCompressed sees:
///   MatMul → Softmax → TopK → ReduceSum → Divide → Transpose → Unsqueeze
/// Returns {unsqueeze_moe, topk_indices}.
static std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>>
build_softmax_routing_for_fuse_test(const ov::Output<ov::Node>& routing_weights, size_t topk) {
    auto softmax = std::make_shared<ov::op::v8::Softmax>(routing_weights, 1);
    auto k = op::v0::Constant::create(element::i32, Shape{}, {static_cast<int32_t>(topk)});
    auto topk_node = std::make_shared<ov::op::v11::TopK>(softmax, k, 1,
        ov::op::v11::TopK::Mode::MAX, ov::op::v11::TopK::SortType::SORT_VALUES);

    auto reduce_axis = op::v0::Constant::create(element::i64, Shape{1}, {1});
    auto reduce_sum = std::make_shared<ov::op::v1::ReduceSum>(topk_node->output(0), reduce_axis, true);
    auto norm = std::make_shared<ov::op::v1::Divide>(topk_node->output(0), reduce_sum);
    auto transpose_order = op::v0::Constant::create(element::i64, Shape{2}, {1, 0});
    auto transpose = std::make_shared<ov::op::v1::Transpose>(norm, transpose_order);
    auto unsqueeze_const = op::v0::Constant::create(element::i64, Shape{1}, {-1});
    auto unsqueeze_moe = std::make_shared<ov::op::v0::Unsqueeze>(transpose, unsqueeze_const);

    return {unsqueeze_moe, topk_node->output(1)};
}

/// Build the post-GatherMatmul sigmoid+bias routing chain that FuseMOE3GemmCompressed sees:
///   MatMul → Sigmoid → Add(bias) → TopK → Convert → GatherElements → ReduceSum → Add(eps) → Divide → Transpose → Unsqueeze
/// Returns {unsqueeze_moe, topk_indices}.
static std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>>
build_sigmoid_routing_for_fuse_test(const ov::Output<ov::Node>& routing_weights,
                                    ov::element::Type data_precision,
                                    size_t number_of_experts,
                                    size_t topk) {
    auto sigmoid = std::make_shared<ov::op::v0::Sigmoid>(routing_weights);
    auto bias = op::v0::Constant::create(data_precision, Shape{1, number_of_experts}, {0.1f});
    auto sig_add = std::make_shared<ov::op::v1::Add>(sigmoid, bias);

    auto k = op::v0::Constant::create(element::i64, Shape{}, {static_cast<int64_t>(topk)});
    auto topk_node = std::make_shared<ov::op::v11::TopK>(sig_add, k, -1,
        ov::op::v11::TopK::Mode::MAX, ov::op::v11::TopK::SortType::SORT_VALUES, ov::element::i64);

    auto convert_topk = std::make_shared<ov::op::v0::Convert>(topk_node->output(1), ov::element::i32);
    auto gather_el = std::make_shared<ov::op::v6::GatherElements>(sigmoid, convert_topk, 1);
    auto reduce_axis = op::v0::Constant::create(element::i64, Shape{1}, {1});
    auto reduce_sum = std::make_shared<ov::op::v1::ReduceSum>(gather_el, reduce_axis, true);
    auto eps = op::v0::Constant::create(data_precision, Shape{1, 1}, {1e-6f});
    auto add_eps = std::make_shared<ov::op::v1::Add>(reduce_sum, eps);
    auto norm = std::make_shared<ov::op::v1::Divide>(gather_el, add_eps);

    auto transpose_order = op::v0::Constant::create(element::i64, Shape{2}, {1, 0});
    auto transpose = std::make_shared<ov::op::v1::Transpose>(norm, transpose_order);
    auto unsqueeze_const = op::v0::Constant::create(element::i64, Shape{1}, {-1});
    auto unsqueeze_moe = std::make_shared<ov::op::v0::Unsqueeze>(transpose, unsqueeze_const);

    return {unsqueeze_moe, convert_topk};
}

TEST_P(FuseMOE3GemmCompressedTest, CompareFunctions) {
    const auto& [routing_type, reshape_on_moe_input] = GetParam();
    {
        // tokens:32, hidden_size:2048, inter_size:768, experts:128, topk:8
        auto hidden_states = std::make_shared<ov::op::v0::Parameter>(element::f16, Shape{4, 8, 2048});
        auto flatten_shape = op::v0::Constant::create(element::i32, Shape{2}, {32, 2048});
        auto hidden_states_reshape = std::make_shared<ov::op::v1::Reshape>(hidden_states, flatten_shape, false);
        auto routers = op::v0::Constant::create(element::f16, Shape{2048, 128}, {0.2});
        auto routing_weights = std::make_shared<ov::op::v0::MatMul>(hidden_states_reshape, routers);

        // Build post-GatherMatmul routing (no ScatterElementsUpdate)
        auto [unsqueeze_moe, topk_indices] =
            routing_type == MoERoutingType::SOFTMAX
                ? build_softmax_routing_for_fuse_test(routing_weights, 8)
                : build_sigmoid_routing_for_fuse_test(routing_weights, element::f16, 128, 8);

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

        auto [unsqueeze_moe, topk_indices] = build_softmax_routing_for_fuse_test(routing_weights, 8);

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

        // Dummy placeholders for SOFTMAX + shared expert (indices 11-12)
        auto dummy_bias = op::v0::Constant::create(element::f16, Shape{1}, {0.0f});
        auto dummy_eps = op::v0::Constant::create(element::f16, Shape{1}, {0.0f});

        ov::intel_gpu::op::MOECompressed::Config config;
        config.hidden_size = 2048;
        config.inter_size = 768;
        config.num_expert = 128;
        config.num_shared_expert = 1;
        config.group_size = 128;
        config.top_k = 8;
        config.out_type = ov::element::f16;
        auto moe_3gemm_fused_compressed = std::make_shared<ov::intel_gpu::op::MOE3GemmFusedCompressed>(
            ov::OutputVector{hidden_states, routing_weights,
                wei_gate, scale_gate, zp_gate, wei_up, scale_up, zp_up, wei_down, scale_down, zp_down,
                dummy_bias, dummy_eps,
                sh_wei_gate, sh_scale_gate, sh_zp_gate, sh_wei_up, sh_scale_up, sh_zp_up,
                sh_wei_down, sh_scale_down, sh_zp_down, sh_gate_gate_wei}, config);

        model_ref = std::make_shared<ov::Model>(moe_3gemm_fused_compressed, ov::ParameterVector{hidden_states});
    }
}

TEST_F(TransformationTestsF, FuseMOE3GemmCompressedTest1) {
    {
        // tokens:32, hidden_size:2048, iter_size:512, experts:512, topk:10
        auto hidden_states = std::make_shared<ov::op::v0::Parameter>(element::f16, Shape{4, 8, 2048});
        auto flatten_shape = op::v0::Constant::create(element::i32, Shape{2}, {32, 2048});
        auto hidden_states_reshape = std::make_shared<ov::op::v1::Reshape>(hidden_states, flatten_shape, false);
        auto routers = op::v0::Constant::create(element::f16, Shape{2048, 512}, {0.2});
        // [32, 512]
        auto routing_weights = std::make_shared<ov::op::v0::MatMul>(hidden_states_reshape, routers);

        // Build post-GatherMatmul softmax routing (no ScatterElementsUpdate)
        auto [unsqueeze_moe, topk_indices] = build_softmax_routing_for_fuse_test(routing_weights, 10);

        // weight
        auto wei_gate = op::v0::Constant::create(element::u4, Shape{512, 512, 16, 128}, {1});
        auto scale_gate = op::v0::Constant::create(element::f16, Shape{512, 16, 512}, {0.01f});
        auto zp_gate = op::v0::Constant::create(element::u4, Shape{512, 16, 512}, {0});
        auto wei_up = op::v0::Constant::create(element::u4, Shape{512, 512, 16, 128}, {1});
        auto scale_up = op::v0::Constant::create(element::f16, Shape{512, 16, 512}, {0.01f});
        auto zp_up = op::v0::Constant::create(element::u4, Shape{512, 16, 512}, {0});
        auto wei_down = op::v0::Constant::create(element::u4, Shape{512, 2048, 4, 128}, {1});
        auto scale_down = op::v0::Constant::create(element::f16, Shape{512, 4, 2048}, {0.01f});
        auto zp_down = op::v0::Constant::create(element::u4, Shape{512, 4, 2048}, {0});

        ov::intel_gpu::op::MOECompressed::Config config;
        config.hidden_size = 2048;
        config.inter_size = 512;
        config.num_expert = 512;
        config.group_size = 128;
        config.top_k = 10;
        config.out_type = ov::element::f16;
        auto moe_compressed = std::make_shared<ov::intel_gpu::op::MOECompressed>(
            ov::OutputVector{hidden_states_reshape, unsqueeze_moe, topk_indices,
                wei_gate, scale_gate, zp_gate, wei_up, scale_up, zp_up, wei_down, scale_down, zp_down}, config);
        model = std::make_shared<ov::Model>(moe_compressed, ov::ParameterVector{hidden_states});
        manager.register_pass<FuseMOE3GemmCompressed>();
    }
    {
        // tokens:32, hidden_size:2048, inter_size:512, experts:512, topk:10
        auto hidden_states = std::make_shared<ov::op::v0::Parameter>(element::f16, Shape{4, 8, 2048});
        auto flatten_shape = op::v0::Constant::create(element::i32, Shape{2}, {32, 2048});
        auto hidden_states_reshape = std::make_shared<ov::op::v1::Reshape>(hidden_states, flatten_shape, false);
        auto routers = op::v0::Constant::create(element::f16, Shape{2048, 512}, {0.2});
        auto routing_weights = std::make_shared<ov::op::v0::MatMul>(hidden_states_reshape, routers);

        // weight
        auto wei_gate = op::v0::Constant::create(element::u4, Shape{512, 512, 16, 128}, {1});
        auto scale_gate = op::v0::Constant::create(element::f16, Shape{512, 16, 512}, {0.01f});
        auto zp_gate = op::v0::Constant::create(element::u4, Shape{512, 16, 512}, {0});
        auto wei_up = op::v0::Constant::create(element::u4, Shape{512, 512, 16, 128}, {1});
        auto scale_up = op::v0::Constant::create(element::f16, Shape{512, 16, 512}, {0.01f});
        auto zp_up = op::v0::Constant::create(element::u4, Shape{512, 16, 512}, {0});
        auto wei_down = op::v0::Constant::create(element::u4, Shape{512, 2048, 4, 128}, {1});
        auto scale_down = op::v0::Constant::create(element::f16, Shape{512, 4, 2048}, {0.01f});
        auto zp_down = op::v0::Constant::create(element::u4, Shape{512, 4, 2048}, {0});

        ov::intel_gpu::op::MOECompressed::Config config;
        config.hidden_size = 2048;
        config.inter_size = 512;
        config.num_expert = 512;
        config.group_size = 128;
        config.top_k = 10;
        config.out_type = ov::element::f16;
        auto moe_3gemm_fused_compressed = std::make_shared<ov::intel_gpu::op::MOE3GemmFusedCompressed>(
            ov::OutputVector{hidden_states_reshape, routing_weights,
                wei_gate, scale_gate, zp_gate, wei_up, scale_up, zp_up, wei_down, scale_down, zp_down}, config);

        model_ref = std::make_shared<ov::Model>(moe_3gemm_fused_compressed, ov::ParameterVector{hidden_states});
    }
}

}  // namespace intel_gpu
}  // namespace test
}  // namespace ov
