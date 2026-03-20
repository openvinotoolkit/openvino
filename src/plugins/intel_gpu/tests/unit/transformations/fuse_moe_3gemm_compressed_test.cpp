#include "openvino/op/constant.hpp"
// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "intel_gpu/op/moe_compressed.hpp"
#include "intel_gpu/op/moe_3gemm_fused_compressed.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
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

        auto softmax = std::make_shared<ov::op::v8::Softmax>(routing_weights, 1);
        auto k = op::v0::Constant::create(element::i32, Shape{}, {8});
        auto topk = std::make_shared<ov::op::v11::TopK>(softmax, k, 1,
            ov::op::v11::TopK::Mode::MAX, ov::op::v11::TopK::SortType::SORT_VALUES);

        // weight output
        auto reduce_axis = op::v0::Constant::create(element::i64, Shape{1}, {1});
        auto reduce_sum = std::make_shared<ov::op::v1::ReduceSum>(topk->output(0), reduce_axis->output(0), true);
        auto norm = std::make_shared<ov::op::v1::Divide>(topk->output(0), reduce_sum->output(0));

        // 32
        auto shape_of = std::make_shared<ov::op::v3::ShapeOf>(topk->output(1));  // [2]{32, 8}
        auto gather_idx = op::v0::Constant::create(element::i64, Shape{}, {0});
        auto gather_axis = op::v0::Constant::create(element::i64, Shape{}, {0});
        auto gather = std::make_shared<ov::op::v8::Gather>(shape_of, gather_idx, gather_axis); // scalar: 32
        auto const_unsqueeze = op::v0::Constant::create(element::i64, Shape{1}, {0});
        auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(gather, const_unsqueeze);  // [1]{32}

        // 128
        auto const0 = op::v0::Constant::create(element::i64, Shape{}, {128});
        auto const1 = op::v0::Constant::create(element::i64, Shape{1}, {0});
        auto unsqueeze1 = std::make_shared<ov::op::v0::Unsqueeze>(const0, const1);  // [1]{128}
        auto concat = std::make_shared<ov::op::v0::Concat>(OutputVector{unsqueeze, unsqueeze1}, 0);  // [2]{32,128}
        auto const3 = op::v0::Constant::create(element::i64, Shape{1}, {1});
        auto concat1 = std::make_shared<ov::op::v0::Concat>(OutputVector{unsqueeze1, unsqueeze, const3}, 0);

        // [32, 128]
        auto zero = op::v0::Constant::create(element::f16, Shape{1}, {0});
        auto bc = std::make_shared<ov::op::v3::Broadcast>(zero, concat);
        auto scatter_axis = op::v0::Constant::create(element::i64, Shape{1}, {1});
        auto scatter = std::make_shared<ov::op::v12::ScatterElementsUpdate>(bc,                  // [32, 128]
                                                                            topk->output(1),     // [32, 8]
                                                                            norm,                // [32, 8]
                                                                            scatter_axis,        // [1]
                                                                            ov::op::v12::ScatterElementsUpdate::Reduction::SUM);
        auto transpose_shape = op::v0::Constant::create(element::i64, Shape{2}, {1, 0});
        auto transpose = std::make_shared<ov::op::v1::Transpose>(scatter, transpose_shape);  // [128, 32]
        auto reshape = std::make_shared<ov::op::v1::Reshape>(transpose, concat1, false);
        auto unsqueeze_const = op::v0::Constant::create(element::i64, Shape{1}, {3});
        auto unsqueeze_moe = std::make_shared<ov::op::v0::Unsqueeze>(reshape, unsqueeze_const); // [128, 1, 32, 1]

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
        config.num_shared_expert = 0;
        config.group_size = 128;
        config.top_k = 8;
        config.out_type = ov::element::f16;
        auto moe_compressed = std::make_shared<ov::intel_gpu::op::MOECompressed>(
            ov::OutputVector{hidden_states, unsqueeze_moe, topk->output(1),
                wei_gate, scale_gate, zp_gate, wei_up, scale_up, zp_up, wei_down, scale_down, zp_down}, config);
        model = std::make_shared<ov::Model>(moe_compressed, ov::ParameterVector{hidden_states});
        manager.register_pass<FuseMOE3GemmCompressed>();
    }
    {
        // tokens:32, hidden_size:2048, iter_size:768, experts:128, topk:8
        auto hidden_states = std::make_shared<ov::op::v0::Parameter>(element::f16, Shape{32, 2048});
        auto routers = op::v0::Constant::create(element::f16, Shape{2048, 128}, {0.2});
        auto routing_weights = std::make_shared<ov::op::v0::MatMul>(hidden_states, routers);

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
        config.num_shared_expert = 0;
        config.group_size = 128;
        config.top_k = 8;
        config.out_type = ov::element::f16;
        auto moe_3gemm_fused_compressed = std::make_shared<ov::intel_gpu::op::MOE3GemmFusedCompressed>(
            ov::OutputVector{hidden_states, routing_weights,
                wei_gate, scale_gate, zp_gate, wei_up, scale_up, zp_up, wei_down, scale_down, zp_down}, config);

        model_ref = std::make_shared<ov::Model>(moe_3gemm_fused_compressed, ov::ParameterVector{hidden_states});
    }
}

}  // namespace intel_gpu
}  // namespace test
}  // namespace ov
