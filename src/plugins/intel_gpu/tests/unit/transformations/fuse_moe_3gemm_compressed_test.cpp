// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/node_builders/moe_builders.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "intel_gpu/op/moe_router_fused.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/gather_elements.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "plugin/transformations/fuse_moe_router.hpp"

using namespace testing;
using namespace ov::intel_gpu;

namespace ov {
namespace test {
namespace intel_gpu {

using namespace ov::test;

class FuseMoERouterTest : public TransformationTestsF,
                          public ::testing::WithParamInterface<MoERoutingType> {
public:
    static std::string get_test_case_name(const ::testing::TestParamInfo<MoERoutingType>& info) {
        switch (info.param) {
        case MoERoutingType::SOFTMAX: return "Softmax";
        case MoERoutingType::SIGMOID_BIAS: return "SigmoidBias";
        default: OPENVINO_THROW("Unsupported routing type");
        }
    }
};

static std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>>
build_softmax_routing_for_fuse_test(const ov::Output<ov::Node>& routing_weights, size_t topk,
                                    std::shared_ptr<ov::op::v0::Constant> per_expert_scale = nullptr) {
    auto softmax = std::make_shared<ov::op::v8::Softmax>(routing_weights, 1);
    auto k = op::v0::Constant::create(element::i32, Shape{}, {static_cast<int32_t>(topk)});
    auto topk_node = std::make_shared<ov::op::v11::TopK>(softmax, k, 1,
        ov::op::v11::TopK::Mode::MAX, ov::op::v11::TopK::SortType::SORT_VALUES);
    auto reduce_axis = op::v0::Constant::create(element::i64, Shape{1}, {1});
    auto reduce_sum = std::make_shared<ov::op::v1::ReduceSum>(topk_node->output(0), reduce_axis, true);
    ov::Output<ov::Node> norm = std::make_shared<ov::op::v1::Divide>(topk_node->output(0), reduce_sum);
    if (per_expert_scale) {
        auto gather_axis = op::v0::Constant::create(element::i32, Shape{}, {0});
        auto gathered = std::make_shared<ov::op::v8::Gather>(per_expert_scale, topk_node->output(1), gather_axis);
        norm = std::make_shared<ov::op::v1::Multiply>(norm, gathered);
    }
    auto transpose_order = op::v0::Constant::create(element::i64, Shape{2}, {1, 0});
    auto transpose = std::make_shared<ov::op::v1::Transpose>(norm, transpose_order);
    auto unsqueeze_const = op::v0::Constant::create(element::i64, Shape{1}, {-1});
    auto unsqueeze_moe = std::make_shared<ov::op::v0::Unsqueeze>(transpose, unsqueeze_const);
    return {unsqueeze_moe, topk_node->output(1)};
}

static std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>>
build_sigmoid_routing_for_fuse_test(const ov::Output<ov::Node>& routing_weights,
                                    ov::element::Type data_precision, size_t number_of_experts, size_t topk,
                                    std::shared_ptr<ov::op::v0::Constant> routed_scale = nullptr) {
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
    ov::Output<ov::Node> norm = std::make_shared<ov::op::v1::Divide>(gather_el, add_eps);
    if (routed_scale) {
        norm = std::make_shared<ov::op::v1::Multiply>(norm, routed_scale);
    }
    auto transpose_order = op::v0::Constant::create(element::i64, Shape{2}, {1, 0});
    auto transpose = std::make_shared<ov::op::v1::Transpose>(norm, transpose_order);
    auto unsqueeze_const = op::v0::Constant::create(element::i64, Shape{1}, {-1});
    auto unsqueeze_moe = std::make_shared<ov::op::v0::Unsqueeze>(transpose, unsqueeze_const);
    return {unsqueeze_moe, convert_topk};
}

TEST_P(FuseMoERouterTest, CompareFunctions) {
    const auto routing_type = GetParam();
    {
        auto hidden_states = std::make_shared<ov::op::v0::Parameter>(element::f16, Shape{4, 8, 2048});
        auto flatten_shape = op::v0::Constant::create(element::i32, Shape{2}, {32, 2048});
        auto hidden_states_reshape = std::make_shared<ov::op::v1::Reshape>(hidden_states, flatten_shape, false);
        auto routers = op::v0::Constant::create(element::f16, Shape{2048, 128}, {0.2});
        auto routing_weights = std::make_shared<ov::op::v0::MatMul>(hidden_states_reshape, routers);

        auto [routing_out, topk_indices] =
            routing_type == MoERoutingType::SOFTMAX
                ? build_softmax_routing_for_fuse_test(routing_weights, 8)
                : build_sigmoid_routing_for_fuse_test(routing_weights, element::f16, 128, 8);

        // Wrap outputs to avoid feeding Result nodes directly (required by replace_output_update_name)
        auto weights_out = std::make_shared<ov::op::v0::Convert>(routing_out, element::f16);
        auto indices_out = std::make_shared<ov::op::v0::Convert>(topk_indices, element::i32);
        model = std::make_shared<ov::Model>(ov::OutputVector{weights_out, indices_out}, ov::ParameterVector{hidden_states});
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

        auto weights_out = std::make_shared<ov::op::v0::Convert>(router_node->output(0), element::f16);
        auto indices_out = std::make_shared<ov::op::v0::Convert>(router_node->output(1), element::i32);
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{weights_out, indices_out}, ov::ParameterVector{hidden_states});
    }
}

INSTANTIATE_TEST_SUITE_P(smoke, FuseMoERouterTest,
    ::testing::Values(MoERoutingType::SOFTMAX, MoERoutingType::SIGMOID_BIAS),
    FuseMoERouterTest::get_test_case_name);

TEST_F(TransformationTestsF, FuseMoERouterDifferentConfigTest) {
    {
        auto hidden_states = std::make_shared<ov::op::v0::Parameter>(element::f16, Shape{4, 8, 2048});
        auto flatten_shape = op::v0::Constant::create(element::i32, Shape{2}, {32, 2048});
        auto hidden_states_reshape = std::make_shared<ov::op::v1::Reshape>(hidden_states, flatten_shape, false);
        auto routers = op::v0::Constant::create(element::f16, Shape{2048, 512}, {0.2});
        auto routing_weights = std::make_shared<ov::op::v0::MatMul>(hidden_states_reshape, routers);
        auto [routing_out, topk_indices] = build_softmax_routing_for_fuse_test(routing_weights, 10);

        auto weights_out = std::make_shared<ov::op::v0::Convert>(routing_out, element::f16);
        auto indices_out = std::make_shared<ov::op::v0::Convert>(topk_indices, element::i32);
        model = std::make_shared<ov::Model>(ov::OutputVector{weights_out, indices_out}, ov::ParameterVector{hidden_states});
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

        auto weights_out = std::make_shared<ov::op::v0::Convert>(router_node->output(0), element::f16);
        auto indices_out = std::make_shared<ov::op::v0::Convert>(router_node->output(1), element::i32);
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{weights_out, indices_out}, ov::ParameterVector{hidden_states});
    }
}

TEST_F(TransformationTestsF, FuseMoERouterSoftmaxPerExpertScaleTest) {
    {
        auto hidden_states = std::make_shared<ov::op::v0::Parameter>(element::f16, Shape{32, 2048});
        auto routers = op::v0::Constant::create(element::f16, Shape{2048, 128}, {0.2});
        auto routing_weights = std::make_shared<ov::op::v0::MatMul>(hidden_states, routers);

        auto per_expert_scale = op::v0::Constant::create(element::f16, Shape{128}, {1.5f});
        auto [routing_out, topk_indices] = build_softmax_routing_for_fuse_test(routing_weights, 8, per_expert_scale);

        auto weights_out = std::make_shared<ov::op::v0::Convert>(routing_out, element::f16);
        model = std::make_shared<ov::Model>(ov::OutputVector{weights_out}, ov::ParameterVector{hidden_states});
        manager.register_pass<FuseMoERouter>();
    }
    {
        auto hidden_states = std::make_shared<ov::op::v0::Parameter>(element::f16, Shape{32, 2048});
        auto routers = op::v0::Constant::create(element::f16, Shape{2048, 128}, {0.2});
        auto routing_weights = std::make_shared<ov::op::v0::MatMul>(hidden_states, routers);

        ov::intel_gpu::op::MoERouterFused::Config router_config;
        router_config.num_expert = 128; router_config.top_k = 8;
        auto router_node = std::make_shared<ov::intel_gpu::op::MoERouterFused>(ov::OutputVector{routing_weights}, router_config);

        auto per_expert_scale = op::v0::Constant::create(element::f16, Shape{128}, {1.5f});
        auto gather_axis = op::v0::Constant::create(element::i32, Shape{}, {0});
        auto gathered = std::make_shared<ov::op::v8::Gather>(per_expert_scale, router_node->output(1), gather_axis);
        auto norm_scaled = std::make_shared<ov::op::v1::Multiply>(router_node->output(0), gathered);

        auto weights_out = std::make_shared<ov::op::v0::Convert>(norm_scaled, element::f16);
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{weights_out}, ov::ParameterVector{hidden_states});
    }
}

TEST_F(TransformationTestsF, FuseMoERouterSigmoidRoutedScaleFactorTest) {
    {
        auto hidden_states = std::make_shared<ov::op::v0::Parameter>(element::f16, Shape{32, 2048});
        auto routers = op::v0::Constant::create(element::f16, Shape{2048, 128}, {0.2});
        auto routing_weights = std::make_shared<ov::op::v0::MatMul>(hidden_states, routers);

        auto scale = op::v0::Constant::create(element::f16, Shape{1, 1}, {2.5f});
        auto [routing_out, topk_indices] = build_sigmoid_routing_for_fuse_test(routing_weights, element::f16, 128, 8, scale);

        auto weights_out = std::make_shared<ov::op::v0::Convert>(routing_out, element::f16);
        auto indices_out = std::make_shared<ov::op::v0::Convert>(topk_indices, element::i32);
        model = std::make_shared<ov::Model>(ov::OutputVector{weights_out, indices_out}, ov::ParameterVector{hidden_states});
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

        auto scale = op::v0::Constant::create(element::f16, Shape{1, 1}, {2.5f});
        auto norm_scaled = std::make_shared<ov::op::v1::Multiply>(router_node->output(0), scale);

        auto weights_out = std::make_shared<ov::op::v0::Convert>(norm_scaled, element::f16);
        auto indices_out = std::make_shared<ov::op::v0::Convert>(router_node->output(1), element::i32);
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{weights_out, indices_out}, ov::ParameterVector{hidden_states});
    }
}

}  // namespace intel_gpu
}  // namespace test
}  // namespace ov
