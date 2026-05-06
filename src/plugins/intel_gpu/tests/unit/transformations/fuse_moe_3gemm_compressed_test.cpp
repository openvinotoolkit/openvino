// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "ov_ops/moe_compressed.hpp"
#include "intel_gpu/op/moe_router_fused.hpp"
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
#include "plugin/transformations/fuse_moe_router.hpp"
#include "common_test_utils/node_builders/moe_builders.hpp"

using namespace testing;
using namespace ov::intel_gpu;

namespace ov {
namespace test {
namespace intel_gpu {

using namespace ov::test;

using FuseMoERouterTestParams = std::tuple<MoERoutingType, bool>;

class FuseMoERouterTest : public TransformationTestsF,
                          public ::testing::WithParamInterface<FuseMoERouterTestParams> {
public:
    static std::string get_test_case_name(const ::testing::TestParamInfo<FuseMoERouterTestParams>& info) {
        std::string name;
        switch (std::get<0>(info.param)) {
        case MoERoutingType::SOFTMAX: name = "Softmax"; break;
        case MoERoutingType::SIGMOID_BIAS: name = "SigmoidBias"; break;
        default: OPENVINO_THROW("Unsupported routing type");
        }
        if (std::get<1>(info.param)) name += "_ReshapeOnMoeInput";
        return name;
    }
};

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

static std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>>
build_sigmoid_routing_for_fuse_test(const ov::Output<ov::Node>& routing_weights,
                                    ov::element::Type data_precision, size_t number_of_experts, size_t topk) {
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

TEST_P(FuseMoERouterTest, CompareFunctions) {
    const auto& [routing_type, reshape_on_moe_input] = GetParam();
    {
        auto hidden_states = std::make_shared<ov::op::v0::Parameter>(element::f16, Shape{4, 8, 2048});
        auto flatten_shape = op::v0::Constant::create(element::i32, Shape{2}, {32, 2048});
        auto hidden_states_reshape = std::make_shared<ov::op::v1::Reshape>(hidden_states, flatten_shape, false);
        auto routers = op::v0::Constant::create(element::f16, Shape{2048, 128}, {0.2});
        auto routing_weights = std::make_shared<ov::op::v0::MatMul>(hidden_states_reshape, routers);

        auto [unsqueeze_moe, topk_indices] =
            routing_type == MoERoutingType::SOFTMAX
                ? build_softmax_routing_for_fuse_test(routing_weights, 8)
                : build_sigmoid_routing_for_fuse_test(routing_weights, element::f16, 128, 8);

        auto wei_gate = op::v0::Constant::create(element::u4, Shape{128, 768, 16, 128}, {1});
        auto scale_gate = op::v0::Constant::create(element::f16, Shape{128, 768, 16}, {0.01f});
        auto zp_gate = op::v0::Constant::create(element::u4, Shape{128, 768, 16}, {0});
        auto wei_up = op::v0::Constant::create(element::u4, Shape{128, 768, 16, 128}, {1});
        auto scale_up = op::v0::Constant::create(element::f16, Shape{128, 768, 16}, {0.01f});
        auto zp_up = op::v0::Constant::create(element::u4, Shape{128, 768, 16}, {0});
        auto wei_down = op::v0::Constant::create(element::u4, Shape{128, 2048, 6, 128}, {1});
        auto scale_down = op::v0::Constant::create(element::f16, Shape{128, 2048, 6}, {0.01f});
        auto zp_down = op::v0::Constant::create(element::u4, Shape{128, 2048, 6}, {0});

        ov::op::internal::MOECompressed::Config config;
        config.expert_type = ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU;
        config.has_zp = true; config.hidden_size = 2048; config.inter_size = 768;
        config.num_expert = 128; config.group_size = 128; config.top_k = 8;
        config.out_type = ov::element::f16;

        auto moe_input_0 = reshape_on_moe_input
            ? ov::Output<ov::Node>(hidden_states_reshape) : ov::Output<ov::Node>(hidden_states);
        auto moe_compressed = std::make_shared<ov::op::internal::MOECompressed>(
            ov::OutputVector{moe_input_0, unsqueeze_moe, topk_indices,
                wei_gate, scale_gate, zp_gate, wei_up, scale_up, zp_up, wei_down, scale_down, zp_down}, config);
        model = std::make_shared<ov::Model>(moe_compressed, ov::ParameterVector{hidden_states});
    }
    manager.register_pass<FuseMoERouter>();
    {
        auto hidden_states = std::make_shared<ov::op::v0::Parameter>(element::f16, Shape{4, 8, 2048});
        auto flatten_shape = op::v0::Constant::create(element::i32, Shape{2}, {32, 2048});
        auto hidden_states_reshape = std::make_shared<ov::op::v1::Reshape>(hidden_states, flatten_shape, false);
        auto routers = op::v0::Constant::create(element::f16, Shape{2048, 128}, {0.2});
        auto routing_weights = std::make_shared<ov::op::v0::MatMul>(hidden_states_reshape, routers);

        ov::intel_gpu::op::MoERouterFused::Config router_config;
        router_config.num_expert = 128; router_config.top_k = 8;
        ov::OutputVector router_args{routing_weights};
        if (routing_type == MoERoutingType::SIGMOID_BIAS) {
            router_config.routing_type = ov::intel_gpu::op::MoERouterFused::RoutingType::SIGMOID_BIAS;
            router_args.push_back(op::v0::Constant::create(element::f16, Shape{1, 128}, {0.1f}));
            router_args.push_back(op::v0::Constant::create(element::f16, Shape{1, 1}, {1e-6f}));
        }
        auto router_node = std::make_shared<ov::intel_gpu::op::MoERouterFused>(router_args, router_config);

        auto wei_gate = op::v0::Constant::create(element::u4, Shape{128, 768, 16, 128}, {1});
        auto scale_gate = op::v0::Constant::create(element::f16, Shape{128, 768, 16}, {0.01f});
        auto zp_gate = op::v0::Constant::create(element::u4, Shape{128, 768, 16}, {0});
        auto wei_up = op::v0::Constant::create(element::u4, Shape{128, 768, 16, 128}, {1});
        auto scale_up = op::v0::Constant::create(element::f16, Shape{128, 768, 16}, {0.01f});
        auto zp_up = op::v0::Constant::create(element::u4, Shape{128, 768, 16}, {0});
        auto wei_down = op::v0::Constant::create(element::u4, Shape{128, 2048, 6, 128}, {1});
        auto scale_down = op::v0::Constant::create(element::f16, Shape{128, 2048, 6}, {0.01f});
        auto zp_down = op::v0::Constant::create(element::u4, Shape{128, 2048, 6}, {0});

        ov::op::internal::MOECompressed::Config config;
        config.expert_type = ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU;
        config.has_zp = true; config.hidden_size = 2048; config.inter_size = 768;
        config.num_expert = 128; config.group_size = 128; config.top_k = 8;
        config.out_type = ov::element::f16;

        auto moe_input_0 = reshape_on_moe_input
            ? ov::Output<ov::Node>(hidden_states_reshape) : ov::Output<ov::Node>(hidden_states);
        auto moe_compressed = std::make_shared<ov::op::internal::MOECompressed>(
            ov::OutputVector{moe_input_0, router_node->output(0), router_node->output(1),
                wei_gate, scale_gate, zp_gate, wei_up, scale_up, zp_up, wei_down, scale_down, zp_down}, config);
        model_ref = std::make_shared<ov::Model>(moe_compressed, ov::ParameterVector{hidden_states});
    }
}

INSTANTIATE_TEST_SUITE_P(smoke, FuseMoERouterTest,
    ::testing::Combine(::testing::Values(MoERoutingType::SOFTMAX, MoERoutingType::SIGMOID_BIAS),
                       ::testing::Values(false, true)),
    FuseMoERouterTest::get_test_case_name);

TEST_F(TransformationTestsF, FuseMoERouterSharedExpertSoftmaxTest) {
    {
        auto hidden_states = std::make_shared<ov::op::v0::Parameter>(element::f16, Shape{32, 2048});
        auto routers = op::v0::Constant::create(element::f16, Shape{2048, 128}, {0.2});
        auto routing_weights = std::make_shared<ov::op::v0::MatMul>(hidden_states, routers);
        auto [unsqueeze_moe, topk_indices] = build_softmax_routing_for_fuse_test(routing_weights, 8);

        auto wei_gate = op::v0::Constant::create(element::u4, Shape{128, 768, 16, 128}, {1});
        auto scale_gate = op::v0::Constant::create(element::f16, Shape{128, 768, 16}, {0.01f});
        auto zp_gate = op::v0::Constant::create(element::u4, Shape{128, 768, 16}, {0});
        auto wei_up = op::v0::Constant::create(element::u4, Shape{128, 768, 16, 128}, {1});
        auto scale_up = op::v0::Constant::create(element::f16, Shape{128, 768, 16}, {0.01f});
        auto zp_up = op::v0::Constant::create(element::u4, Shape{128, 768, 16}, {0});
        auto wei_down = op::v0::Constant::create(element::u4, Shape{128, 2048, 6, 128}, {1});
        auto scale_down = op::v0::Constant::create(element::f16, Shape{128, 2048, 6}, {0.01f});
        auto zp_down = op::v0::Constant::create(element::u4, Shape{128, 2048, 6}, {0});
        auto sh_wei_gate = op::v0::Constant::create(element::u4, Shape{1, 768, 16, 128}, {2});
        auto sh_scale_gate = op::v0::Constant::create(element::f16, Shape{1, 768, 16}, {0.02f});
        auto sh_zp_gate = op::v0::Constant::create(element::u4, Shape{1, 768, 16}, {0});
        auto sh_wei_up = op::v0::Constant::create(element::u4, Shape{1, 768, 16, 128}, {2});
        auto sh_scale_up = op::v0::Constant::create(element::f16, Shape{1, 768, 16}, {0.02f});
        auto sh_zp_up = op::v0::Constant::create(element::u4, Shape{1, 768, 16}, {0});
        auto sh_wei_down = op::v0::Constant::create(element::u4, Shape{1, 2048, 6, 128}, {2});
        auto sh_scale_down = op::v0::Constant::create(element::f16, Shape{1, 2048, 6}, {0.02f});
        auto sh_zp_down = op::v0::Constant::create(element::u4, Shape{1, 2048, 6}, {0});
        auto sh_gate_gate_wei = op::v0::Constant::create(element::f16, Shape{2048, 1}, {0.5f});

        ov::op::internal::MOECompressed::Config config;
        config.expert_type = ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU;
        config.has_zp = true; config.hidden_size = 2048; config.inter_size = 768;
        config.num_expert = 128; config.num_shared_expert = 1; config.group_size = 128;
        config.top_k = 8; config.out_type = ov::element::f16;
        auto moe_compressed = std::make_shared<ov::op::internal::MOECompressed>(
            ov::OutputVector{hidden_states, unsqueeze_moe, topk_indices,
                wei_gate, scale_gate, zp_gate, wei_up, scale_up, zp_up, wei_down, scale_down, zp_down,
                sh_wei_gate, sh_scale_gate, sh_zp_gate, sh_wei_up, sh_scale_up, sh_zp_up,
                sh_wei_down, sh_scale_down, sh_zp_down, sh_gate_gate_wei}, config);
        model = std::make_shared<ov::Model>(moe_compressed, ov::ParameterVector{hidden_states});
        manager.register_pass<FuseMoERouter>();
    }
    {
        auto hidden_states = std::make_shared<ov::op::v0::Parameter>(element::f16, Shape{32, 2048});
        auto routers = op::v0::Constant::create(element::f16, Shape{2048, 128}, {0.2});
        auto routing_weights = std::make_shared<ov::op::v0::MatMul>(hidden_states, routers);

        ov::intel_gpu::op::MoERouterFused::Config router_config;
        router_config.num_expert = 128; router_config.top_k = 8;
        auto router_node = std::make_shared<ov::intel_gpu::op::MoERouterFused>(ov::OutputVector{routing_weights}, router_config);

        auto wei_gate = op::v0::Constant::create(element::u4, Shape{128, 768, 16, 128}, {1});
        auto scale_gate = op::v0::Constant::create(element::f16, Shape{128, 768, 16}, {0.01f});
        auto zp_gate = op::v0::Constant::create(element::u4, Shape{128, 768, 16}, {0});
        auto wei_up = op::v0::Constant::create(element::u4, Shape{128, 768, 16, 128}, {1});
        auto scale_up = op::v0::Constant::create(element::f16, Shape{128, 768, 16}, {0.01f});
        auto zp_up = op::v0::Constant::create(element::u4, Shape{128, 768, 16}, {0});
        auto wei_down = op::v0::Constant::create(element::u4, Shape{128, 2048, 6, 128}, {1});
        auto scale_down = op::v0::Constant::create(element::f16, Shape{128, 2048, 6}, {0.01f});
        auto zp_down = op::v0::Constant::create(element::u4, Shape{128, 2048, 6}, {0});
        auto sh_wei_gate = op::v0::Constant::create(element::u4, Shape{1, 768, 16, 128}, {2});
        auto sh_scale_gate = op::v0::Constant::create(element::f16, Shape{1, 768, 16}, {0.02f});
        auto sh_zp_gate = op::v0::Constant::create(element::u4, Shape{1, 768, 16}, {0});
        auto sh_wei_up = op::v0::Constant::create(element::u4, Shape{1, 768, 16, 128}, {2});
        auto sh_scale_up = op::v0::Constant::create(element::f16, Shape{1, 768, 16}, {0.02f});
        auto sh_zp_up = op::v0::Constant::create(element::u4, Shape{1, 768, 16}, {0});
        auto sh_wei_down = op::v0::Constant::create(element::u4, Shape{1, 2048, 6, 128}, {2});
        auto sh_scale_down = op::v0::Constant::create(element::f16, Shape{1, 2048, 6}, {0.02f});
        auto sh_zp_down = op::v0::Constant::create(element::u4, Shape{1, 2048, 6}, {0});
        auto sh_gate_gate_wei = op::v0::Constant::create(element::f16, Shape{2048, 1}, {0.5f});

        ov::op::internal::MOECompressed::Config config;
        config.expert_type = ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU;
        config.has_zp = true; config.hidden_size = 2048; config.inter_size = 768;
        config.num_expert = 128; config.num_shared_expert = 1; config.group_size = 128;
        config.top_k = 8; config.out_type = ov::element::f16;
        auto moe_fused = std::make_shared<ov::op::internal::MOECompressed>(
            ov::OutputVector{hidden_states, router_node->output(0), router_node->output(1),
                wei_gate, scale_gate, zp_gate, wei_up, scale_up, zp_up, wei_down, scale_down, zp_down,
                sh_wei_gate, sh_scale_gate, sh_zp_gate, sh_wei_up, sh_scale_up, sh_zp_up,
                sh_wei_down, sh_scale_down, sh_zp_down, sh_gate_gate_wei}, config);
        model_ref = std::make_shared<ov::Model>(moe_fused, ov::ParameterVector{hidden_states});
    }
}

TEST_F(TransformationTestsF, FuseMoERouterSharedExpertSigmoidTest) {
    {
        auto hidden_states = std::make_shared<ov::op::v0::Parameter>(element::f16, Shape{32, 2048});
        auto routers = op::v0::Constant::create(element::f16, Shape{2048, 128}, {0.2});
        auto routing_weights = std::make_shared<ov::op::v0::MatMul>(hidden_states, routers);
        auto [unsqueeze_moe, topk_indices] = build_sigmoid_routing_for_fuse_test(routing_weights, element::f16, 128, 8);

        auto wei_gate = op::v0::Constant::create(element::u4, Shape{128, 768, 16, 128}, {1});
        auto scale_gate = op::v0::Constant::create(element::f16, Shape{128, 768, 16}, {0.01f});
        auto zp_gate = op::v0::Constant::create(element::u4, Shape{128, 768, 16}, {0});
        auto wei_up = op::v0::Constant::create(element::u4, Shape{128, 768, 16, 128}, {1});
        auto scale_up = op::v0::Constant::create(element::f16, Shape{128, 768, 16}, {0.01f});
        auto zp_up = op::v0::Constant::create(element::u4, Shape{128, 768, 16}, {0});
        auto wei_down = op::v0::Constant::create(element::u4, Shape{128, 2048, 6, 128}, {1});
        auto scale_down = op::v0::Constant::create(element::f16, Shape{128, 2048, 6}, {0.01f});
        auto zp_down = op::v0::Constant::create(element::u4, Shape{128, 2048, 6}, {0});
        auto sh_wei_gate = op::v0::Constant::create(element::u4, Shape{1, 768, 16, 128}, {2});
        auto sh_scale_gate = op::v0::Constant::create(element::f16, Shape{1, 768, 16}, {0.02f});
        auto sh_zp_gate = op::v0::Constant::create(element::u4, Shape{1, 768, 16}, {0});
        auto sh_wei_up = op::v0::Constant::create(element::u4, Shape{1, 768, 16, 128}, {2});
        auto sh_scale_up = op::v0::Constant::create(element::f16, Shape{1, 768, 16}, {0.02f});
        auto sh_zp_up = op::v0::Constant::create(element::u4, Shape{1, 768, 16}, {0});
        auto sh_wei_down = op::v0::Constant::create(element::u4, Shape{1, 2048, 6, 128}, {2});
        auto sh_scale_down = op::v0::Constant::create(element::f16, Shape{1, 2048, 6}, {0.02f});
        auto sh_zp_down = op::v0::Constant::create(element::u4, Shape{1, 2048, 6}, {0});
        auto sh_gate_gate_wei = op::v0::Constant::create(element::f16, Shape{2048, 1}, {0.5f});

        ov::op::internal::MOECompressed::Config config;
        config.expert_type = ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU;
        config.has_zp = true; config.hidden_size = 2048; config.inter_size = 768;
        config.num_expert = 128; config.num_shared_expert = 1; config.group_size = 128;
        config.top_k = 8; config.out_type = ov::element::f16;
        auto moe_compressed = std::make_shared<ov::op::internal::MOECompressed>(
            ov::OutputVector{hidden_states, unsqueeze_moe, topk_indices,
                wei_gate, scale_gate, zp_gate, wei_up, scale_up, zp_up, wei_down, scale_down, zp_down,
                sh_wei_gate, sh_scale_gate, sh_zp_gate, sh_wei_up, sh_scale_up, sh_zp_up,
                sh_wei_down, sh_scale_down, sh_zp_down, sh_gate_gate_wei}, config);
        model = std::make_shared<ov::Model>(moe_compressed, ov::ParameterVector{hidden_states});
        manager.register_pass<FuseMoERouter>();
    }
    {
        auto hidden_states = std::make_shared<ov::op::v0::Parameter>(element::f16, Shape{32, 2048});
        auto routers = op::v0::Constant::create(element::f16, Shape{2048, 128}, {0.2});
        auto routing_weights = std::make_shared<ov::op::v0::MatMul>(hidden_states, routers);

        ov::intel_gpu::op::MoERouterFused::Config router_config;
        router_config.num_expert = 128; router_config.top_k = 8;
        router_config.routing_type = ov::intel_gpu::op::MoERouterFused::RoutingType::SIGMOID_BIAS;
        auto routing_bias = op::v0::Constant::create(element::f16, Shape{1, 128}, {0.1f});
        auto routing_eps = op::v0::Constant::create(element::f16, Shape{1, 1}, {1e-6f});
        auto router_node = std::make_shared<ov::intel_gpu::op::MoERouterFused>(
            ov::OutputVector{routing_weights, routing_bias, routing_eps}, router_config);

        auto wei_gate = op::v0::Constant::create(element::u4, Shape{128, 768, 16, 128}, {1});
        auto scale_gate = op::v0::Constant::create(element::f16, Shape{128, 768, 16}, {0.01f});
        auto zp_gate = op::v0::Constant::create(element::u4, Shape{128, 768, 16}, {0});
        auto wei_up = op::v0::Constant::create(element::u4, Shape{128, 768, 16, 128}, {1});
        auto scale_up = op::v0::Constant::create(element::f16, Shape{128, 768, 16}, {0.01f});
        auto zp_up = op::v0::Constant::create(element::u4, Shape{128, 768, 16}, {0});
        auto wei_down = op::v0::Constant::create(element::u4, Shape{128, 2048, 6, 128}, {1});
        auto scale_down = op::v0::Constant::create(element::f16, Shape{128, 2048, 6}, {0.01f});
        auto zp_down = op::v0::Constant::create(element::u4, Shape{128, 2048, 6}, {0});
        auto sh_wei_gate = op::v0::Constant::create(element::u4, Shape{1, 768, 16, 128}, {2});
        auto sh_scale_gate = op::v0::Constant::create(element::f16, Shape{1, 768, 16}, {0.02f});
        auto sh_zp_gate = op::v0::Constant::create(element::u4, Shape{1, 768, 16}, {0});
        auto sh_wei_up = op::v0::Constant::create(element::u4, Shape{1, 768, 16, 128}, {2});
        auto sh_scale_up = op::v0::Constant::create(element::f16, Shape{1, 768, 16}, {0.02f});
        auto sh_zp_up = op::v0::Constant::create(element::u4, Shape{1, 768, 16}, {0});
        auto sh_wei_down = op::v0::Constant::create(element::u4, Shape{1, 2048, 6, 128}, {2});
        auto sh_scale_down = op::v0::Constant::create(element::f16, Shape{1, 2048, 6}, {0.02f});
        auto sh_zp_down = op::v0::Constant::create(element::u4, Shape{1, 2048, 6}, {0});
        auto sh_gate_gate_wei = op::v0::Constant::create(element::f16, Shape{2048, 1}, {0.5f});

        ov::op::internal::MOECompressed::Config config;
        config.expert_type = ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU;
        config.has_zp = true; config.hidden_size = 2048; config.inter_size = 768;
        config.num_expert = 128; config.num_shared_expert = 1; config.group_size = 128;
        config.top_k = 8; config.out_type = ov::element::f16;
        auto moe_fused = std::make_shared<ov::op::internal::MOECompressed>(
            ov::OutputVector{hidden_states, router_node->output(0), router_node->output(1),
                wei_gate, scale_gate, zp_gate, wei_up, scale_up, zp_up, wei_down, scale_down, zp_down,
                sh_wei_gate, sh_scale_gate, sh_zp_gate, sh_wei_up, sh_scale_up, sh_zp_up,
                sh_wei_down, sh_scale_down, sh_zp_down, sh_gate_gate_wei}, config);
        model_ref = std::make_shared<ov::Model>(moe_fused, ov::ParameterVector{hidden_states});
    }
}

TEST_F(TransformationTestsF, FuseMoERouterDifferentConfigTest) {
    {
        auto hidden_states = std::make_shared<ov::op::v0::Parameter>(element::f16, Shape{4, 8, 2048});
        auto flatten_shape = op::v0::Constant::create(element::i32, Shape{2}, {32, 2048});
        auto hidden_states_reshape = std::make_shared<ov::op::v1::Reshape>(hidden_states, flatten_shape, false);
        auto routers = op::v0::Constant::create(element::f16, Shape{2048, 512}, {0.2});
        auto routing_weights = std::make_shared<ov::op::v0::MatMul>(hidden_states_reshape, routers);
        auto [unsqueeze_moe, topk_indices] = build_softmax_routing_for_fuse_test(routing_weights, 10);

        auto wei_gate = op::v0::Constant::create(element::u4, Shape{512, 512, 16, 128}, {1});
        auto scale_gate = op::v0::Constant::create(element::f16, Shape{512, 512, 16}, {0.01f});
        auto zp_gate = op::v0::Constant::create(element::u4, Shape{512, 512, 16}, {0});
        auto wei_up = op::v0::Constant::create(element::u4, Shape{512, 512, 16, 128}, {1});
        auto scale_up = op::v0::Constant::create(element::f16, Shape{512, 512, 16}, {0.01f});
        auto zp_up = op::v0::Constant::create(element::u4, Shape{512, 512, 16}, {0});
        auto wei_down = op::v0::Constant::create(element::u4, Shape{512, 2048, 4, 128}, {1});
        auto scale_down = op::v0::Constant::create(element::f16, Shape{512, 2048, 4}, {0.01f});
        auto zp_down = op::v0::Constant::create(element::u4, Shape{512, 2048, 4}, {0});

        ov::op::internal::MOECompressed::Config config;
        config.expert_type = ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU;
        config.has_zp = true; config.hidden_size = 2048; config.inter_size = 512;
        config.num_expert = 512; config.group_size = 128; config.top_k = 10;
        config.out_type = ov::element::f16;
        auto moe_compressed = std::make_shared<ov::op::internal::MOECompressed>(
            ov::OutputVector{hidden_states_reshape, unsqueeze_moe, topk_indices,
                wei_gate, scale_gate, zp_gate, wei_up, scale_up, zp_up, wei_down, scale_down, zp_down}, config);
        model = std::make_shared<ov::Model>(moe_compressed, ov::ParameterVector{hidden_states});
        manager.register_pass<FuseMoERouter>();
    }
    {
        auto hidden_states = std::make_shared<ov::op::v0::Parameter>(element::f16, Shape{4, 8, 2048});
        auto flatten_shape = op::v0::Constant::create(element::i32, Shape{2}, {32, 2048});
        auto hidden_states_reshape = std::make_shared<ov::op::v1::Reshape>(hidden_states, flatten_shape, false);
        auto routers = op::v0::Constant::create(element::f16, Shape{2048, 512}, {0.2});
        auto routing_weights = std::make_shared<ov::op::v0::MatMul>(hidden_states_reshape, routers);

        ov::intel_gpu::op::MoERouterFused::Config router_config;
        router_config.num_expert = 512; router_config.top_k = 10;
        auto router_node = std::make_shared<ov::intel_gpu::op::MoERouterFused>(ov::OutputVector{routing_weights}, router_config);

        auto wei_gate = op::v0::Constant::create(element::u4, Shape{512, 512, 16, 128}, {1});
        auto scale_gate = op::v0::Constant::create(element::f16, Shape{512, 512, 16}, {0.01f});
        auto zp_gate = op::v0::Constant::create(element::u4, Shape{512, 512, 16}, {0});
        auto wei_up = op::v0::Constant::create(element::u4, Shape{512, 512, 16, 128}, {1});
        auto scale_up = op::v0::Constant::create(element::f16, Shape{512, 512, 16}, {0.01f});
        auto zp_up = op::v0::Constant::create(element::u4, Shape{512, 512, 16}, {0});
        auto wei_down = op::v0::Constant::create(element::u4, Shape{512, 2048, 4, 128}, {1});
        auto scale_down = op::v0::Constant::create(element::f16, Shape{512, 2048, 4}, {0.01f});
        auto zp_down = op::v0::Constant::create(element::u4, Shape{512, 2048, 4}, {0});

        ov::op::internal::MOECompressed::Config config;
        config.expert_type = ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU;
        config.has_zp = true; config.hidden_size = 2048; config.inter_size = 512;
        config.num_expert = 512; config.group_size = 128; config.top_k = 10;
        config.out_type = ov::element::f16;
        auto moe_fused = std::make_shared<ov::op::internal::MOECompressed>(
            ov::OutputVector{hidden_states_reshape, router_node->output(0), router_node->output(1),
                wei_gate, scale_gate, zp_gate, wei_up, scale_up, zp_up, wei_down, scale_down, zp_down}, config);
        model_ref = std::make_shared<ov::Model>(moe_fused, ov::ParameterVector{hidden_states});
    }
}

}  // namespace intel_gpu
}  // namespace test
}  // namespace ov
