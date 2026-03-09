// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "intel_gpu/op/moe_compressed.hpp"
#include "intel_gpu/op/moe_3gemm_fused_compressed.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/subtract.hpp"
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
class FuseMOE3GemmCompressedTest : public TransformationTestsF, public ::testing::WithParamInterface<MoERoutingType> {
public:
    static std::string get_test_case_name(const ::testing::TestParamInfo<MoERoutingType>& info) {
        switch (info.param) {
        case MoERoutingType::SOFTMAX:
            return "Softmax";
        case MoERoutingType::SIGMOID_BIAS:
            return "SigmoidBias";
        default:
            OPENVINO_THROW("Unsupported routing type");
        }
    }
};

TEST_P(FuseMOE3GemmCompressedTest, CompareFunctions) {
    const auto routing_type = GetParam();
    {
        // tokens:32, hidden_size:2048, inter_size:768, experts:128, topk:8
        auto hidden_states = std::make_shared<ov::op::v0::Parameter>(element::f16, Shape{4, 8, 2048});
        auto flatten_shape = op::v0::Constant::create(element::i32, Shape{2}, {32, 2048});
        auto hidden_states_reshape = std::make_shared<ov::op::v1::Reshape>(hidden_states, flatten_shape, false);
        auto routers = op::v0::Constant::create(element::f16, Shape{2048, 128}, {0.2});
        auto routing_weights = std::make_shared<ov::op::v0::MatMul>(hidden_states_reshape, routers);

        // tokens:32, num_experts:128, topk:8
        auto [unsqueeze_moe, topk_indices] =
            routing_type == MoERoutingType::SOFTMAX
                ? build_softmax_routing_subgraph(routing_weights, 128, 8)
                : build_sigmoid_bias_routing_subgraph(routing_weights, element::f16, 128, 8);

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
        auto moe_compressed = std::make_shared<ov::intel_gpu::op::MOECompressed>(
            ov::OutputVector{hidden_states_reshape, unsqueeze_moe, topk_indices,
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

        auto moe_3gemm_fused_compressed = std::make_shared<ov::intel_gpu::op::MOE3GemmFusedCompressed>(args, config);
        model_ref = std::make_shared<ov::Model>(moe_3gemm_fused_compressed, ov::ParameterVector{hidden_states});
    }
}

INSTANTIATE_TEST_SUITE_P(smoke,
                         FuseMOE3GemmCompressedTest,
                         ::testing::Values(MoERoutingType::SOFTMAX, MoERoutingType::SIGMOID_BIAS),
                         FuseMOE3GemmCompressedTest::get_test_case_name);

TEST_F(TransformationTestsF, FuseMOE3GemmCompressedTest1) {
    {
        // tokens:32, hidden_size:2048, iter_size:512, experts:512, topk:10
        auto hidden_states = std::make_shared<ov::op::v0::Parameter>(element::f16, Shape{4, 8, 2048});
        auto flatten_shape = op::v0::Constant::create(element::i32, Shape{2}, {32, 2048});
        auto hidden_states_reshape = std::make_shared<ov::op::v1::Reshape>(hidden_states, flatten_shape, false);
        auto routers = op::v0::Constant::create(element::f16, Shape{2048, 512}, {0.2});
        // [32, 512]
        auto routing_weights = std::make_shared<ov::op::v0::MatMul>(hidden_states_reshape, routers);

        auto softmax = std::make_shared<ov::op::v8::Softmax>(routing_weights, 1);
        auto k = op::v0::Constant::create(element::i64, Shape{}, {10});
        // [32, 10]
        auto topk = std::make_shared<ov::op::v11::TopK>(softmax, k, 1,
            ov::op::v11::TopK::Mode::MAX, ov::op::v11::TopK::SortType::SORT_VALUES);

        // weight output
        auto reduce_axis = op::v0::Constant::create(element::i64, Shape{1}, {1});
        auto reduce_sum = std::make_shared<ov::op::v1::ReduceSum>(topk->output(0), reduce_axis->output(0), true);
        auto norm = std::make_shared<ov::op::v1::Divide>(topk->output(0), reduce_sum->output(0));

        // index
        auto topk_indices_i32 = std::make_shared<ov::op::v0::Convert>(topk->output(1), ov::element::i32);
        auto shape_of = std::make_shared<ov::op::v3::ShapeOf>(topk_indices_i32);

        auto slice = std::make_shared<ov::op::v8::Slice>(norm,
                                                         op::v0::Constant::create(element::i32, Shape{2}, {0, 0}),
                                                         shape_of,
                                                         op::v0::Constant::create(element::i32, Shape{2}, {1, 1}),
                                                         op::v0::Constant::create(element::i32, Shape{2}, {0, 1}));

        auto zero = op::v0::Constant::create(element::f16, Shape{1}, {0});
        auto router_shape = op::v0::Constant::create(element::i64, Shape{2}, {32, 512});
        auto bc = std::make_shared<ov::op::v3::Broadcast>(zero, router_shape);
        auto scatter_axis = op::v0::Constant::create(element::i64, Shape{1}, {1});
        auto scatter = std::make_shared<ov::op::v12::ScatterElementsUpdate>(bc,
                                                                            topk_indices_i32,
                                                                            slice,
                                                                            scatter_axis);
        // [512, 32]
        auto transpose_shape = op::v0::Constant::create(element::i64, Shape{2}, {1, 0});
        auto transpose = std::make_shared<ov::op::v1::Transpose>(scatter, transpose_shape);
        // [512, 4, 8]
        auto router_shape_transpose = op::v0::Constant::create(element::i64, Shape{3}, {512, 4, 8});
        auto reshape = std::make_shared<ov::op::v1::Reshape>(transpose, router_shape_transpose, false);
        // [512, 4, 8, 1]
        auto unsqueeze_const = op::v0::Constant::create(element::i64, Shape{1}, {3});
        auto unsqueeze_moe = std::make_shared<ov::op::v0::Unsqueeze>(reshape, unsqueeze_const);

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
            ov::OutputVector{hidden_states, unsqueeze_moe, topk->output(1),
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
        auto hidden_state_shape = std::make_shared<ov::op::v3::ShapeOf>(hidden_states);
        auto reshape_back = std::make_shared<ov::op::v1::Reshape>(moe_3gemm_fused_compressed->output(0), hidden_state_shape, false);

        model_ref = std::make_shared<ov::Model>(reshape_back, ov::ParameterVector{hidden_states});
    }
}

}  // namespace intel_gpu
}  // namespace test
}  // namespace ov
