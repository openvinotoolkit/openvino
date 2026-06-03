// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/transformations/fuse_moe_router.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/node_builders/moe_builders.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "intel_gpu/op/moe_router_fused.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather_elements.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/op/unsqueeze.hpp"

using namespace testing;
using namespace ov::intel_gpu;

namespace ov {
namespace test {
namespace intel_gpu {

using namespace ov::test;

using TestParams = std::tuple<MoERoutingType, bool, size_t, size_t>;  // routing_type, with_convert_on_indices, num_expert, top_k

class FuseMoERouterTest : public TransformationTestsF, public ::testing::WithParamInterface<TestParams> {
public:
    static std::string get_test_case_name(const ::testing::TestParamInfo<TestParams>& info) {
        const auto routing_type = std::get<0>(info.param);
        const bool with_convert = std::get<1>(info.param);
        const size_t num_expert = std::get<2>(info.param);
        const size_t top_k = std::get<3>(info.param);
        std::string name;
        switch (routing_type) {
        case MoERoutingType::SOFTMAX:
            name = "Softmax";
            break;
        case MoERoutingType::SIGMOID_BIAS:
            name = "SigmoidBias";
            break;
        default:
            OPENVINO_THROW("Unsupported routing type");
        }
        name += with_convert ? "_WithConvert" : "_NoConvert";
        name += "_E" + std::to_string(num_expert) + "_K" + std::to_string(top_k);
        return name;
    }
};

static std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>> build_softmax_routing_for_fuse_test(const ov::Output<ov::Node>& routing_weights,
                                                                                                 size_t topk,
                                                                                                 bool with_convert_on_indices) {
    auto softmax = std::make_shared<ov::op::v8::Softmax>(routing_weights, 1);
    const auto index_type = with_convert_on_indices ? ov::element::i64 : ov::element::i32;
    auto k = op::v0::Constant::create(element::i32, Shape{}, {static_cast<int32_t>(topk)});
    auto topk_node = std::make_shared<ov::op::v11::TopK>(softmax, k, 1, ov::op::v11::TopK::Mode::MAX, ov::op::v11::TopK::SortType::SORT_VALUES, index_type);
    auto reduce_axis = op::v0::Constant::create(element::i64, Shape{1}, {1});
    auto reduce_sum = std::make_shared<ov::op::v1::ReduceSum>(topk_node->output(0), reduce_axis, true);
    ov::Output<ov::Node> norm = std::make_shared<ov::op::v1::Divide>(topk_node->output(0), reduce_sum);
    ov::Output<ov::Node> indices = topk_node->output(1);
    if (with_convert_on_indices)
        indices = std::make_shared<ov::op::v0::Convert>(indices, ov::element::i32);
    return {norm, indices};
}

static std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>> build_sigmoid_routing_for_fuse_test(const ov::Output<ov::Node>& routing_weights,
                                                                                                 ov::element::Type data_precision,
                                                                                                 size_t number_of_experts,
                                                                                                 size_t topk,
                                                                                                 bool with_convert_on_indices) {
    auto sigmoid = std::make_shared<ov::op::v0::Sigmoid>(routing_weights);
    auto bias = op::v0::Constant::create(data_precision, Shape{1, number_of_experts}, {0.1f});
    auto sig_add = std::make_shared<ov::op::v1::Add>(sigmoid, bias);
    const auto index_type = with_convert_on_indices ? ov::element::i64 : ov::element::i32;
    auto k = op::v0::Constant::create(element::i64, Shape{}, {static_cast<int64_t>(topk)});
    auto topk_node = std::make_shared<ov::op::v11::TopK>(sig_add, k, -1, ov::op::v11::TopK::Mode::MAX, ov::op::v11::TopK::SortType::SORT_VALUES, index_type);
    ov::Output<ov::Node> indices = topk_node->output(1);
    if (with_convert_on_indices)
        indices = std::make_shared<ov::op::v0::Convert>(indices, ov::element::i32);
    auto gather_el = std::make_shared<ov::op::v6::GatherElements>(sigmoid, indices, 1);
    auto reduce_axis = op::v0::Constant::create(element::i64, Shape{1}, {1});
    auto reduce_sum = std::make_shared<ov::op::v1::ReduceSum>(gather_el, reduce_axis, true);
    auto eps = op::v0::Constant::create(data_precision, Shape{1, 1}, {1e-6f});
    auto add_eps = std::make_shared<ov::op::v1::Add>(reduce_sum, eps);
    ov::Output<ov::Node> norm = std::make_shared<ov::op::v1::Divide>(gather_el, add_eps);
    return {norm, indices};
}

TEST_P(FuseMoERouterTest, CompareFunctions) {
    const auto [routing_type, with_convert_on_indices, num_expert, top_k] = GetParam();
    {
        auto hidden_states = std::make_shared<ov::op::v0::Parameter>(element::f16, Shape{4, 8, 2048});
        auto flatten_shape = op::v0::Constant::create(element::i32, Shape{2}, {32, 2048});
        auto hidden_states_reshape = std::make_shared<ov::op::v1::Reshape>(hidden_states, flatten_shape, false);
        auto routers = op::v0::Constant::create(element::f16, Shape{2048, num_expert}, {0.2});
        auto routing_weights = std::make_shared<ov::op::v0::MatMul>(hidden_states_reshape, routers);

        const auto [routing_out, topk_indices] =
            routing_type == MoERoutingType::SOFTMAX
                ? build_softmax_routing_for_fuse_test(routing_weights, top_k, with_convert_on_indices)
                : build_sigmoid_routing_for_fuse_test(routing_weights, element::f16, num_expert, top_k, with_convert_on_indices);

        // Wrap outputs to avoid feeding Result nodes directly (required by replace_output_update_name)
        auto weights_out = std::make_shared<ov::op::v0::Unsqueeze>(routing_out, ov::op::v0::Constant::create(element::i32, Shape{1}, {0}));
        auto indices_out = std::make_shared<ov::op::v0::Unsqueeze>(topk_indices, ov::op::v0::Constant::create(element::i32, Shape{1}, {0}));
        model = std::make_shared<ov::Model>(ov::OutputVector{weights_out, indices_out}, ov::ParameterVector{hidden_states});
    }
    manager.register_pass<FuseMoERouter>();
    {
        auto hidden_states = std::make_shared<ov::op::v0::Parameter>(element::f16, Shape{4, 8, 2048});
        auto flatten_shape = op::v0::Constant::create(element::i32, Shape{2}, {32, 2048});
        auto hidden_states_reshape = std::make_shared<ov::op::v1::Reshape>(hidden_states, flatten_shape, false);
        auto routers = op::v0::Constant::create(element::f16, Shape{2048, num_expert}, {0.2});
        auto routing_weights = std::make_shared<ov::op::v0::MatMul>(hidden_states_reshape, routers);

        ov::intel_gpu::op::MoERouterFused::Config router_config;
        router_config.num_expert = num_expert;
        router_config.top_k = top_k;
        ov::OutputVector router_args{routing_weights};
        if (routing_type == MoERoutingType::SIGMOID_BIAS) {
            router_config.routing_type = ov::intel_gpu::op::MoERouterFused::RoutingType::SIGMOID_BIAS;
            router_args.push_back(op::v0::Constant::create(element::f16, Shape{1, num_expert}, {0.1f}));
            router_args.push_back(op::v0::Constant::create(element::f16, Shape{1, 1}, {1e-6f}));
        }
        auto router_node = std::make_shared<ov::intel_gpu::op::MoERouterFused>(router_args, router_config);

        auto weights_out = std::make_shared<ov::op::v0::Unsqueeze>(router_node->output(0), ov::op::v0::Constant::create(element::i32, Shape{1}, {0}));
        auto indices_out = std::make_shared<ov::op::v0::Unsqueeze>(router_node->output(1), ov::op::v0::Constant::create(element::i32, Shape{1}, {0}));
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{weights_out, indices_out}, ov::ParameterVector{hidden_states});
    }
}

INSTANTIATE_TEST_SUITE_P(smoke,
                         FuseMoERouterTest,
                         ::testing::Combine(::testing::Values(MoERoutingType::SOFTMAX, MoERoutingType::SIGMOID_BIAS),
                                            ::testing::Values(true, false),
                                            ::testing::Values(128, 512),
                                            ::testing::Values(8, 10)),
                         FuseMoERouterTest::get_test_case_name);

}  // namespace intel_gpu
}  // namespace test
}  // namespace ov
