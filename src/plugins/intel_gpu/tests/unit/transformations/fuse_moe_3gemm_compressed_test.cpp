// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/transformations/fuse_moe_3gemm_compressed.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/node_builders/moe_builders.hpp"
#include "common_test_utils/ov_test_utils.hpp"
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
#include "openvino/op/softmax.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "ov_ops/moe_compressed.hpp"

using namespace testing;
using namespace ov::intel_gpu;

namespace ov {
namespace test {
namespace intel_gpu {

using namespace ov::test;
using AT = ov::op::internal::MOE::Activation_type;

using FuseMOE3GemmCompressedTestParams = std::tuple<MoERoutingType,
                                                    bool,  // reshape_on_moe_input
                                                    bool,  // with_routed_scale
                                                    ov::op::internal::MOE::Activation_type>;

class FuseMOE3GemmCompressedTest : public TransformationTestsF, public ::testing::WithParamInterface<FuseMOE3GemmCompressedTestParams> {
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
        if (std::get<2>(info.param))
            name += "_RoutedScalingFactor";
        switch (std::get<3>(info.param)) {
        case AT::SWIGLU:
            break;
        case AT::GEGLU_TANH:
            name += "_GeluTanh";
            break;
        case AT::GEGLU_ERF:
            name += "_GeluErf";
            break;
        }
        return name;
    }
};

// Softmax routing: MatMul → Softmax → TopK → ReduceSum → Divide
//   [→ Convert(i32) → Gather(per_expert_scale, topk_idx) → Multiply(norm, gathered)  when per_expert_scale is set]
//   → Transpose → Unsqueeze.
static std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>> build_softmax_routing_for_fuse_test(const ov::Output<ov::Node>& routing_weights,
                                                                                                 size_t topk,
                                                                                                 size_t number_of_experts = 0,
                                                                                                 std::optional<float> per_expert_scale = std::nullopt) {
    auto softmax = std::make_shared<ov::op::v8::Softmax>(routing_weights, 1);
    auto k = op::v0::Constant::create(element::i32, Shape{}, {static_cast<int32_t>(topk)});
    auto topk_node = std::make_shared<ov::op::v11::TopK>(softmax, k, 1, ov::op::v11::TopK::Mode::MAX, ov::op::v11::TopK::SortType::SORT_VALUES);

    auto reduce_axis = op::v0::Constant::create(element::i64, Shape{1}, {1});
    auto reduce_sum = std::make_shared<ov::op::v1::ReduceSum>(topk_node->output(0), reduce_axis, true);
    ov::Output<ov::Node> norm = std::make_shared<ov::op::v1::Divide>(topk_node->output(0), reduce_sum);

    ov::Output<ov::Node> topk_indices = topk_node->output(1);
    if (per_expert_scale.has_value()) {
        auto convert_topk = std::make_shared<ov::op::v0::Convert>(topk_node->output(1), element::i32);
        auto scale_const = op::v0::Constant::create(routing_weights.get_element_type(), Shape{number_of_experts}, {*per_expert_scale});
        auto gather_axis = op::v0::Constant::create(element::i32, Shape{}, {0});
        auto gathered = std::make_shared<ov::op::v8::Gather>(scale_const, convert_topk, gather_axis, 0);
        norm = std::make_shared<ov::op::v1::Multiply>(norm, gathered);
        topk_indices = convert_topk;
    }

    auto transpose_order = op::v0::Constant::create(element::i64, Shape{2}, {1, 0});
    auto transpose = std::make_shared<ov::op::v1::Transpose>(norm, transpose_order);
    auto unsqueeze_const = op::v0::Constant::create(element::i64, Shape{1}, {-1});
    auto unsqueeze_moe = std::make_shared<ov::op::v0::Unsqueeze>(transpose, unsqueeze_const);

    return {unsqueeze_moe, topk_indices};
}

// Sigmoid+bias routing: Sigmoid → Add → TopK → Convert → GatherElements → ReduceSum → Add(eps) → Divide
//                       [→ Multiply(routed_scaling_factor) when scale_value is set]
//                       → Transpose → Unsqueeze.
static std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>> build_sigmoid_routing_for_fuse_test(const ov::Output<ov::Node>& routing_weights,
                                                                                                 ov::element::Type data_precision,
                                                                                                 size_t number_of_experts,
                                                                                                 size_t topk,
                                                                                                 std::optional<float> scale_value = std::nullopt) {
    auto sigmoid = std::make_shared<ov::op::v0::Sigmoid>(routing_weights);
    auto bias = op::v0::Constant::create(data_precision, Shape{1, number_of_experts}, {0.1f});
    auto sig_add = std::make_shared<ov::op::v1::Add>(sigmoid, bias);

    auto k = op::v0::Constant::create(element::i64, Shape{}, {static_cast<int64_t>(topk)});
    auto topk_node =
        std::make_shared<ov::op::v11::TopK>(sig_add, k, -1, ov::op::v11::TopK::Mode::MAX, ov::op::v11::TopK::SortType::SORT_VALUES, ov::element::i64);

    auto convert_topk = std::make_shared<ov::op::v0::Convert>(topk_node->output(1), ov::element::i32);
    auto gather_el = std::make_shared<ov::op::v6::GatherElements>(sigmoid, convert_topk, 1);
    auto reduce_axis = op::v0::Constant::create(element::i64, Shape{1}, {1});
    auto reduce_sum = std::make_shared<ov::op::v1::ReduceSum>(gather_el, reduce_axis, true);
    auto eps = op::v0::Constant::create(data_precision, Shape{1, 1}, {1e-6f});
    auto add_eps = std::make_shared<ov::op::v1::Add>(reduce_sum, eps);
    std::shared_ptr<ov::Node> norm = std::make_shared<ov::op::v1::Divide>(gather_el, add_eps);

    // Mirror trinity-mini afmoe `routed_scaling_factor`: an extra Multiply between
    // the normalized routing weights and the final Transpose. The matcher's optional<Multiply>
    // absorbs it; the callback re-applies it as a post-op Multiply on the fused MOE output.
    if (scale_value.has_value()) {
        auto scale_const = op::v0::Constant::create(data_precision, Shape{1, 1}, {*scale_value});
        norm = std::make_shared<ov::op::v1::Multiply>(norm, scale_const);
    }

    auto transpose_order = op::v0::Constant::create(element::i64, Shape{2}, {1, 0});
    auto transpose = std::make_shared<ov::op::v1::Transpose>(norm, transpose_order);
    auto unsqueeze_const = op::v0::Constant::create(element::i64, Shape{1}, {-1});
    auto unsqueeze_moe = std::make_shared<ov::op::v0::Unsqueeze>(transpose, unsqueeze_const);

    return {unsqueeze_moe, convert_topk};
}

TEST_P(FuseMOE3GemmCompressedTest, CompareFunctions) {
    const auto& [routing_type, reshape_on_moe_input, with_routed_scale, activation_type] = GetParam();
    constexpr float routed_scale_value = 2.5f;
    // SOFTMAX uses per-expert scale (Gather+Multiply, folded into scale_down in the reference).
    // SIGMOID_BIAS uses a global routed_scaling_factor Multiply (re-emitted as a post-op in the reference).
    constexpr auto data_precision = element::f32;

    // ── MoE shape parameters (shared between input model and reference model) ──
    constexpr size_t batch = 4;
    constexpr size_t seq_len = 8;
    constexpr size_t tokens = batch * seq_len;
    constexpr size_t hidden_size = 2048;
    constexpr size_t inter_size = 768;
    constexpr size_t num_experts = 128;
    constexpr size_t top_k = 8;
    constexpr size_t group_size = 128;
    constexpr size_t gate_up_groups = hidden_size / group_size;  // gate/up weights grouped along K=hidden_size
    constexpr size_t down_groups = inter_size / group_size;      // down weight grouped along K=inter_size

    {
        auto hidden_states = std::make_shared<ov::op::v0::Parameter>(data_precision, Shape{batch, seq_len, hidden_size});
        auto flatten_shape = op::v0::Constant::create(element::i32, Shape{2}, {static_cast<int32_t>(tokens), static_cast<int32_t>(hidden_size)});
        auto hidden_states_reshape = std::make_shared<ov::op::v1::Reshape>(hidden_states, flatten_shape, false);
        auto routers = op::v0::Constant::create(data_precision, Shape{hidden_size, num_experts}, {0.2});

        // Gemma-4 style: the router MatMul uses a separately-normed hidden state while
        // MOECompressed input[0] uses the plain reshape. Only applies to SOFTMAX+per-expert scale.
        ov::Output<ov::Node> router_input = hidden_states_reshape;
        if (routing_type == MoERoutingType::SOFTMAX && with_routed_scale) {
            auto norm_scale = op::v0::Constant::create(data_precision, Shape{1, 1, hidden_size}, {1.0f});
            auto hidden_normed = std::make_shared<ov::op::v1::Multiply>(hidden_states, norm_scale);
            router_input = std::make_shared<ov::op::v1::Reshape>(hidden_normed, flatten_shape, false);
        }
        auto routing_weights = std::make_shared<ov::op::v0::MatMul>(router_input, routers);
        const std::optional<float> softmax_per_expert_scale =
            (routing_type == MoERoutingType::SOFTMAX && with_routed_scale) ? std::optional<float>{routed_scale_value} : std::nullopt;
        const std::optional<float> sigmoid_scale =
            (routing_type == MoERoutingType::SIGMOID_BIAS && with_routed_scale) ? std::optional<float>{routed_scale_value} : std::nullopt;
        auto [unsqueeze_moe, topk_indices] = routing_type == MoERoutingType::SOFTMAX
                                                 ? build_softmax_routing_for_fuse_test(routing_weights, top_k, num_experts, softmax_per_expert_scale)
                                                 : build_sigmoid_routing_for_fuse_test(routing_weights, data_precision, num_experts, top_k, sigmoid_scale);

        // Compressed weights: 4-D weight = {experts, N, num_groups, group_size};
        // 3-D scale/zp = {experts, N, num_groups}. K = num_groups * group_size.
        auto wei_gate = op::v0::Constant::create(element::u4, Shape{num_experts, inter_size, gate_up_groups, group_size}, {1});
        auto scale_gate = op::v0::Constant::create(element::f16, Shape{num_experts, inter_size, gate_up_groups}, {0.01f});
        auto zp_gate = op::v0::Constant::create(element::u4, Shape{num_experts, inter_size, gate_up_groups}, {0});
        auto wei_up = op::v0::Constant::create(element::u4, Shape{num_experts, inter_size, gate_up_groups, group_size}, {1});
        auto scale_up = op::v0::Constant::create(element::f16, Shape{num_experts, inter_size, gate_up_groups}, {0.01f});
        auto zp_up = op::v0::Constant::create(element::u4, Shape{num_experts, inter_size, gate_up_groups}, {0});
        auto wei_down = op::v0::Constant::create(element::u4, Shape{num_experts, hidden_size, down_groups, group_size}, {1});
        auto scale_down = op::v0::Constant::create(element::f16, Shape{num_experts, hidden_size, down_groups}, {0.01f});
        auto zp_down = op::v0::Constant::create(element::u4, Shape{num_experts, hidden_size, down_groups}, {0});

        ov::op::internal::MOECompressed::Config config;
        config.expert_type = ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU;
        config.activation_type = activation_type;
        config.has_zp = true;
        config.hidden_size = hidden_size;
        config.inter_size = inter_size;
        config.num_expert = num_experts;
        config.group_size = group_size;
        config.top_k = top_k;
        config.out_type = ov::element::f16;

        // When reshape_on_moe_input is true, MOECompressed gets the flattened 2D hidden_states_reshape;
        // otherwise it gets the original 3D hidden_states.
        auto moe_input_0 = reshape_on_moe_input ? ov::Output<ov::Node>(hidden_states_reshape) : ov::Output<ov::Node>(hidden_states);
        auto moe_compressed = std::make_shared<ov::op::internal::MOECompressed>(
            ov::OutputVector{moe_input_0, unsqueeze_moe, topk_indices, wei_gate, scale_gate, zp_gate, wei_up, scale_up, zp_up, wei_down, scale_down, zp_down},
            config);
        model = std::make_shared<ov::Model>(moe_compressed, ov::ParameterVector{hidden_states});
    }
    manager.register_pass<FuseMOE3GemmCompressed>();
    {
        auto hidden_states = std::make_shared<ov::op::v0::Parameter>(data_precision, Shape{batch, seq_len, hidden_size});
        auto flatten_shape = op::v0::Constant::create(element::i32, Shape{2}, {static_cast<int32_t>(tokens), static_cast<int32_t>(hidden_size)});
        auto hidden_states_reshape = std::make_shared<ov::op::v1::Reshape>(hidden_states, flatten_shape, false);
        auto routers = op::v0::Constant::create(data_precision, Shape{hidden_size, num_experts}, {0.2});

        // Mirror the normed router input from the input model.
        ov::Output<ov::Node> router_input = hidden_states_reshape;
        if (routing_type == MoERoutingType::SOFTMAX && with_routed_scale) {
            auto norm_scale = op::v0::Constant::create(data_precision, Shape{1, 1, hidden_size}, {1.0f});
            auto hidden_normed = std::make_shared<ov::op::v1::Multiply>(hidden_states, norm_scale);
            router_input = std::make_shared<ov::op::v1::Reshape>(hidden_normed, flatten_shape, false);
        }
        auto routing_weights = std::make_shared<ov::op::v0::MatMul>(router_input, routers);

        auto wei_gate = op::v0::Constant::create(element::u4, Shape{num_experts, inter_size, gate_up_groups, group_size}, {1});
        auto scale_gate = op::v0::Constant::create(element::f16, Shape{num_experts, inter_size, gate_up_groups}, {0.01f});
        auto zp_gate = op::v0::Constant::create(element::u4, Shape{num_experts, inter_size, gate_up_groups}, {0});
        auto wei_up = op::v0::Constant::create(element::u4, Shape{num_experts, inter_size, gate_up_groups, group_size}, {1});
        auto scale_up = op::v0::Constant::create(element::f16, Shape{num_experts, inter_size, gate_up_groups}, {0.01f});
        auto zp_up = op::v0::Constant::create(element::u4, Shape{num_experts, inter_size, gate_up_groups}, {0});
        auto wei_down = op::v0::Constant::create(element::u4, Shape{num_experts, hidden_size, down_groups, group_size}, {1});
        // SOFTMAX + with_routed_scale: per-expert scale is folded into scale_down by the transformation.
        const float scale_down_val = (routing_type == MoERoutingType::SOFTMAX && with_routed_scale) ? 0.01f * routed_scale_value : 0.01f;
        auto scale_down = op::v0::Constant::create(element::f16, Shape{num_experts, hidden_size, down_groups}, {scale_down_val});
        auto zp_down = op::v0::Constant::create(element::u4, Shape{num_experts, hidden_size, down_groups}, {0});

        ov::op::internal::MOECompressed::Config config;
        config.expert_type = ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU;
        config.activation_type = activation_type;
        config.has_zp = true;
        config.hidden_size = hidden_size;
        config.inter_size = inter_size;
        config.num_expert = num_experts;
        config.group_size = group_size;
        config.top_k = top_k;
        config.out_type = ov::element::f16;
        if (routing_type == MoERoutingType::SOFTMAX) {
            config.routing_type = ov::op::internal::MOECompressed::RoutingType::SOFTMAX;
        } else if (routing_type == MoERoutingType::SIGMOID_BIAS) {
            config.routing_type = ov::op::internal::MOECompressed::RoutingType::SIGMOID_BIAS;
        } else {
            OPENVINO_THROW("Unsupported routing type");
        }

        // When SOFTMAX+with_routed_scale and no reshape on MOE input, the transformation binds
        // hidden_state_reshape (optional) to hidden_states (3D, a Parameter) because the normed
        // router uses its own separate Reshape. So hs_reshaped = hidden_states, and
        // moe_compressed->input_value(0) == hs_reshaped => no reshape-back is emitted.
        const bool normed_router_no_reshape = (routing_type == MoERoutingType::SOFTMAX && with_routed_scale && !reshape_on_moe_input);
        ov::Output<ov::Node> fused_hs_input = normed_router_no_reshape ? ov::Output<ov::Node>(hidden_states) : ov::Output<ov::Node>(hidden_states_reshape);
        ov::OutputVector args{fused_hs_input, routing_weights, wei_gate, scale_gate, zp_gate, wei_up, scale_up, zp_up, wei_down, scale_down, zp_down};
        if (routing_type == MoERoutingType::SIGMOID_BIAS) {
            auto routing_bias = op::v0::Constant::create(data_precision, Shape{1, num_experts}, {0.1f});
            args.push_back(routing_bias);
            auto routing_eps = op::v0::Constant::create(data_precision, Shape{1, 1}, {1e-6f});
            args.push_back(routing_eps);
        }

        std::shared_ptr<ov::Node> result = std::make_shared<ov::intel_gpu::op::MOE3GemmFusedCompressed>(args, config);

        // When reshape_on_moe_input is false AND the router uses hidden_states_reshape (not a separate
        // normed reshape), hs_reshaped = hidden_states_reshape != moe_compressed->input_value(0), so the
        // transformation inserts a reshape-back. For the normed router path (SOFTMAX+with_routed_scale),
        // hs_reshaped = hidden_states (3D) == moe_compressed->input_value(0), so no reshape-back.
        if (!reshape_on_moe_input && !normed_router_no_reshape) {
            auto hidden_state_shape = std::make_shared<ov::op::v3::ShapeOf>(hidden_states);
            result = std::make_shared<ov::op::v1::Reshape>(result, hidden_state_shape, false);
        }

        // SIGMOID_BIAS: the transformation re-applies the absorbed routed_scaling_factor as a post-op Multiply.
        // When the scale dtype (data_precision=f32) != MOE output dtype (f16), the transformation
        // inserts a Convert so the Multiply inputs have matching types. Mirror that here.
        // SOFTMAX: the per-expert scale is folded into scale_down above; no post-op Multiply.
        if (with_routed_scale && routing_type == MoERoutingType::SIGMOID_BIAS) {
            auto post_scale = op::v0::Constant::create(data_precision, Shape{1, 1}, {routed_scale_value});
            auto post_scale_converted = std::make_shared<ov::op::v0::Convert>(post_scale, element::f16);
            result = std::make_shared<ov::op::v1::Multiply>(result, post_scale_converted);
        }

        model_ref = std::make_shared<ov::Model>(result, ov::ParameterVector{hidden_states});
    }
}

INSTANTIATE_TEST_SUITE_P(smoke,
                         FuseMOE3GemmCompressedTest,
                         ::testing::Combine(::testing::Values(MoERoutingType::SOFTMAX, MoERoutingType::SIGMOID_BIAS),
                                            ::testing::Bool(),  // reshape_on_moe_input
                                            ::testing::Bool(),  // with_routed_scale (per-expert for SOFTMAX, global Multiply for SIGMOID_BIAS)
                                            ::testing::Values(AT::SWIGLU, AT::GEGLU_TANH, AT::GEGLU_ERF)),
                         FuseMOE3GemmCompressedTest::get_test_case_name);

TEST_F(TransformationTestsF, FuseMOE3GemmSharedExpertCompressedTest) {
    // ── MoE shape parameters (shared between input model and reference model) ──
    constexpr size_t tokens = 32;
    constexpr size_t hidden_size = 2048;
    constexpr size_t inter_size = 768;
    constexpr size_t num_experts = 128;
    constexpr size_t num_shared = 1;
    constexpr size_t top_k = 8;
    constexpr size_t group_size = 128;
    constexpr size_t gate_up_groups = hidden_size / group_size;  // gate/up grouped along K=hidden_size
    constexpr size_t down_groups = inter_size / group_size;      // down grouped along K=inter_size

    {
        auto hidden_states = std::make_shared<ov::op::v0::Parameter>(element::f16, Shape{tokens, hidden_size});
        auto routers = op::v0::Constant::create(element::f16, Shape{hidden_size, num_experts}, {0.2});
        auto routing_weights = std::make_shared<ov::op::v0::MatMul>(hidden_states, routers);

        auto [unsqueeze_moe, topk_indices] = build_softmax_routing_for_fuse_test(routing_weights, top_k);

        auto wei_gate = op::v0::Constant::create(element::u4, Shape{num_experts, inter_size, gate_up_groups, group_size}, {1});
        auto scale_gate = op::v0::Constant::create(element::f16, Shape{num_experts, inter_size, gate_up_groups}, {0.01f});
        auto zp_gate = op::v0::Constant::create(element::u4, Shape{num_experts, inter_size, gate_up_groups}, {0});
        auto wei_up = op::v0::Constant::create(element::u4, Shape{num_experts, inter_size, gate_up_groups, group_size}, {1});
        auto scale_up = op::v0::Constant::create(element::f16, Shape{num_experts, inter_size, gate_up_groups}, {0.01f});
        auto zp_up = op::v0::Constant::create(element::u4, Shape{num_experts, inter_size, gate_up_groups}, {0});
        auto wei_down = op::v0::Constant::create(element::u4, Shape{num_experts, hidden_size, down_groups, group_size}, {1});
        auto scale_down = op::v0::Constant::create(element::f16, Shape{num_experts, hidden_size, down_groups}, {0.01f});
        auto zp_down = op::v0::Constant::create(element::u4, Shape{num_experts, hidden_size, down_groups}, {0});

        // Shared expert weights: same N/K layout, leading dim = num_shared.
        auto sh_wei_gate = op::v0::Constant::create(element::u4, Shape{num_shared, inter_size, gate_up_groups, group_size}, {2});
        auto sh_scale_gate = op::v0::Constant::create(element::f16, Shape{num_shared, inter_size, gate_up_groups}, {0.02f});
        auto sh_zp_gate = op::v0::Constant::create(element::u4, Shape{num_shared, inter_size, gate_up_groups}, {0});
        auto sh_wei_up = op::v0::Constant::create(element::u4, Shape{num_shared, inter_size, gate_up_groups, group_size}, {2});
        auto sh_scale_up = op::v0::Constant::create(element::f16, Shape{num_shared, inter_size, gate_up_groups}, {0.02f});
        auto sh_zp_up = op::v0::Constant::create(element::u4, Shape{num_shared, inter_size, gate_up_groups}, {0});
        auto sh_wei_down = op::v0::Constant::create(element::u4, Shape{num_shared, hidden_size, down_groups, group_size}, {2});
        auto sh_scale_down = op::v0::Constant::create(element::f16, Shape{num_shared, hidden_size, down_groups}, {0.02f});
        auto sh_zp_down = op::v0::Constant::create(element::u4, Shape{num_shared, hidden_size, down_groups}, {0});
        auto sh_gate_gate_wei = op::v0::Constant::create(element::f16, Shape{hidden_size, num_shared}, {0.5f});

        ov::op::internal::MOECompressed::Config config;
        config.expert_type = ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU;
        config.has_zp = true;
        config.hidden_size = hidden_size;
        config.inter_size = inter_size;
        config.num_expert = num_experts;
        config.num_shared_expert = num_shared;
        config.group_size = group_size;
        config.top_k = top_k;
        config.out_type = ov::element::f16;
        auto moe_compressed = std::make_shared<ov::op::internal::MOECompressed>(
            ov::OutputVector{hidden_states, unsqueeze_moe, topk_indices, wei_gate,      scale_gate,  zp_gate,         wei_up,     scale_up,
                             zp_up,         wei_down,      scale_down,   zp_down,       sh_wei_gate, sh_scale_gate,   sh_zp_gate, sh_wei_up,
                             sh_scale_up,   sh_zp_up,      sh_wei_down,  sh_scale_down, sh_zp_down,  sh_gate_gate_wei},
            config);
        model = std::make_shared<ov::Model>(moe_compressed, ov::ParameterVector{hidden_states});
        manager.register_pass<FuseMOE3GemmCompressed>();
    }
    {
        auto hidden_states = std::make_shared<ov::op::v0::Parameter>(element::f16, Shape{tokens, hidden_size});
        auto routers = op::v0::Constant::create(element::f16, Shape{hidden_size, num_experts}, {0.2});
        auto routing_weights = std::make_shared<ov::op::v0::MatMul>(hidden_states, routers);

        // MOE expert weights
        auto wei_gate = op::v0::Constant::create(element::u4, Shape{num_experts, inter_size, gate_up_groups, group_size}, {1});
        auto scale_gate = op::v0::Constant::create(element::f16, Shape{num_experts, inter_size, gate_up_groups}, {0.01f});
        auto zp_gate = op::v0::Constant::create(element::u4, Shape{num_experts, inter_size, gate_up_groups}, {0});
        auto wei_up = op::v0::Constant::create(element::u4, Shape{num_experts, inter_size, gate_up_groups, group_size}, {1});
        auto scale_up = op::v0::Constant::create(element::f16, Shape{num_experts, inter_size, gate_up_groups}, {0.01f});
        auto zp_up = op::v0::Constant::create(element::u4, Shape{num_experts, inter_size, gate_up_groups}, {0});
        auto wei_down = op::v0::Constant::create(element::u4, Shape{num_experts, hidden_size, down_groups, group_size}, {1});
        auto scale_down = op::v0::Constant::create(element::f16, Shape{num_experts, hidden_size, down_groups}, {0.01f});
        auto zp_down = op::v0::Constant::create(element::u4, Shape{num_experts, hidden_size, down_groups}, {0});

        // Shared expert weights: same N/K layout, leading dim = num_shared.
        auto sh_wei_gate = op::v0::Constant::create(element::u4, Shape{num_shared, inter_size, gate_up_groups, group_size}, {2});
        auto sh_scale_gate = op::v0::Constant::create(element::f16, Shape{num_shared, inter_size, gate_up_groups}, {0.02f});
        auto sh_zp_gate = op::v0::Constant::create(element::u4, Shape{num_shared, inter_size, gate_up_groups}, {0});
        auto sh_wei_up = op::v0::Constant::create(element::u4, Shape{num_shared, inter_size, gate_up_groups, group_size}, {2});
        auto sh_scale_up = op::v0::Constant::create(element::f16, Shape{num_shared, inter_size, gate_up_groups}, {0.02f});
        auto sh_zp_up = op::v0::Constant::create(element::u4, Shape{num_shared, inter_size, gate_up_groups}, {0});
        auto sh_wei_down = op::v0::Constant::create(element::u4, Shape{num_shared, hidden_size, down_groups, group_size}, {2});
        auto sh_scale_down = op::v0::Constant::create(element::f16, Shape{num_shared, hidden_size, down_groups}, {0.02f});
        auto sh_zp_down = op::v0::Constant::create(element::u4, Shape{num_shared, hidden_size, down_groups}, {0});
        auto sh_gate_gate_wei = op::v0::Constant::create(element::f16, Shape{hidden_size, num_shared}, {0.5f});

        // Dummy placeholders for SOFTMAX + shared expert (indices 11-12)
        auto dummy_bias = op::v0::Constant::create(element::f16, Shape{1}, {0.0f});
        auto dummy_eps = op::v0::Constant::create(element::f16, Shape{1}, {0.0f});

        ov::op::internal::MOECompressed::Config config;
        config.expert_type = ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU;
        config.has_zp = true;
        config.hidden_size = hidden_size;
        config.inter_size = inter_size;
        config.num_expert = num_experts;
        config.num_shared_expert = num_shared;
        config.group_size = group_size;
        config.top_k = top_k;
        config.out_type = ov::element::f16;
        auto moe_3gemm_fused_compressed = std::make_shared<ov::intel_gpu::op::MOE3GemmFusedCompressed>(
            ov::OutputVector{hidden_states, routing_weights, wei_gate, scale_gate,  zp_gate,       wei_up,      scale_up,        zp_up,
                             wei_down,      scale_down,      zp_down,  dummy_bias,  dummy_eps,     sh_wei_gate, sh_scale_gate,   sh_zp_gate,
                             sh_wei_up,     sh_scale_up,     sh_zp_up, sh_wei_down, sh_scale_down, sh_zp_down,  sh_gate_gate_wei},
            config);

        model_ref = std::make_shared<ov::Model>(moe_3gemm_fused_compressed, ov::ParameterVector{hidden_states});
    }
}

TEST_F(TransformationTestsF, FuseMOE3GemmSharedExpertCompressedSigmoidTest) {
    // ── MoE shape parameters (shared between input model and reference model) ──
    constexpr size_t tokens = 32;
    constexpr size_t hidden_size = 2048;
    constexpr size_t inter_size = 768;
    constexpr size_t num_experts = 128;
    constexpr size_t num_shared = 1;
    constexpr size_t top_k = 8;
    constexpr size_t group_size = 128;
    constexpr size_t gate_up_groups = hidden_size / group_size;  // gate/up grouped along K=hidden_size
    constexpr size_t down_groups = inter_size / group_size;      // down grouped along K=inter_size

    {
        auto hidden_states = std::make_shared<ov::op::v0::Parameter>(element::f16, Shape{tokens, hidden_size});
        auto routers = op::v0::Constant::create(element::f16, Shape{hidden_size, num_experts}, {0.2});
        auto routing_weights = std::make_shared<ov::op::v0::MatMul>(hidden_states, routers);

        auto [unsqueeze_moe, topk_indices] = build_sigmoid_routing_for_fuse_test(routing_weights, element::f16, num_experts, top_k);

        auto wei_gate = op::v0::Constant::create(element::u4, Shape{num_experts, inter_size, gate_up_groups, group_size}, {1});
        auto scale_gate = op::v0::Constant::create(element::f16, Shape{num_experts, inter_size, gate_up_groups}, {0.01f});
        auto zp_gate = op::v0::Constant::create(element::u4, Shape{num_experts, inter_size, gate_up_groups}, {0});
        auto wei_up = op::v0::Constant::create(element::u4, Shape{num_experts, inter_size, gate_up_groups, group_size}, {1});
        auto scale_up = op::v0::Constant::create(element::f16, Shape{num_experts, inter_size, gate_up_groups}, {0.01f});
        auto zp_up = op::v0::Constant::create(element::u4, Shape{num_experts, inter_size, gate_up_groups}, {0});
        auto wei_down = op::v0::Constant::create(element::u4, Shape{num_experts, hidden_size, down_groups, group_size}, {1});
        auto scale_down = op::v0::Constant::create(element::f16, Shape{num_experts, hidden_size, down_groups}, {0.01f});
        auto zp_down = op::v0::Constant::create(element::u4, Shape{num_experts, hidden_size, down_groups}, {0});

        // Shared expert weights: same N/K layout, leading dim = num_shared.
        auto sh_wei_gate = op::v0::Constant::create(element::u4, Shape{num_shared, inter_size, gate_up_groups, group_size}, {2});
        auto sh_scale_gate = op::v0::Constant::create(element::f16, Shape{num_shared, inter_size, gate_up_groups}, {0.02f});
        auto sh_zp_gate = op::v0::Constant::create(element::u4, Shape{num_shared, inter_size, gate_up_groups}, {0});
        auto sh_wei_up = op::v0::Constant::create(element::u4, Shape{num_shared, inter_size, gate_up_groups, group_size}, {2});
        auto sh_scale_up = op::v0::Constant::create(element::f16, Shape{num_shared, inter_size, gate_up_groups}, {0.02f});
        auto sh_zp_up = op::v0::Constant::create(element::u4, Shape{num_shared, inter_size, gate_up_groups}, {0});
        auto sh_wei_down = op::v0::Constant::create(element::u4, Shape{num_shared, hidden_size, down_groups, group_size}, {2});
        auto sh_scale_down = op::v0::Constant::create(element::f16, Shape{num_shared, hidden_size, down_groups}, {0.02f});
        auto sh_zp_down = op::v0::Constant::create(element::u4, Shape{num_shared, hidden_size, down_groups}, {0});
        auto sh_gate_gate_wei = op::v0::Constant::create(element::f16, Shape{hidden_size, num_shared}, {0.5f});

        ov::op::internal::MOECompressed::Config config;
        config.expert_type = ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU;
        config.has_zp = true;
        config.hidden_size = hidden_size;
        config.inter_size = inter_size;
        config.num_expert = num_experts;
        config.num_shared_expert = num_shared;
        config.group_size = group_size;
        config.top_k = top_k;
        config.out_type = ov::element::f16;
        auto moe_compressed = std::make_shared<ov::op::internal::MOECompressed>(
            ov::OutputVector{hidden_states, unsqueeze_moe, topk_indices, wei_gate,      scale_gate,  zp_gate,         wei_up,     scale_up,
                             zp_up,         wei_down,      scale_down,   zp_down,       sh_wei_gate, sh_scale_gate,   sh_zp_gate, sh_wei_up,
                             sh_scale_up,   sh_zp_up,      sh_wei_down,  sh_scale_down, sh_zp_down,  sh_gate_gate_wei},
            config);
        model = std::make_shared<ov::Model>(moe_compressed, ov::ParameterVector{hidden_states});
        manager.register_pass<FuseMOE3GemmCompressed>();
    }
    {
        auto hidden_states = std::make_shared<ov::op::v0::Parameter>(element::f16, Shape{tokens, hidden_size});
        auto routers = op::v0::Constant::create(element::f16, Shape{hidden_size, num_experts}, {0.2});
        auto routing_weights = std::make_shared<ov::op::v0::MatMul>(hidden_states, routers);

        // MOE expert weights
        auto wei_gate = op::v0::Constant::create(element::u4, Shape{num_experts, inter_size, gate_up_groups, group_size}, {1});
        auto scale_gate = op::v0::Constant::create(element::f16, Shape{num_experts, inter_size, gate_up_groups}, {0.01f});
        auto zp_gate = op::v0::Constant::create(element::u4, Shape{num_experts, inter_size, gate_up_groups}, {0});
        auto wei_up = op::v0::Constant::create(element::u4, Shape{num_experts, inter_size, gate_up_groups, group_size}, {1});
        auto scale_up = op::v0::Constant::create(element::f16, Shape{num_experts, inter_size, gate_up_groups}, {0.01f});
        auto zp_up = op::v0::Constant::create(element::u4, Shape{num_experts, inter_size, gate_up_groups}, {0});
        auto wei_down = op::v0::Constant::create(element::u4, Shape{num_experts, hidden_size, down_groups, group_size}, {1});
        auto scale_down = op::v0::Constant::create(element::f16, Shape{num_experts, hidden_size, down_groups}, {0.01f});
        auto zp_down = op::v0::Constant::create(element::u4, Shape{num_experts, hidden_size, down_groups}, {0});

        // Shared expert weights: same N/K layout, leading dim = num_shared.
        auto sh_wei_gate = op::v0::Constant::create(element::u4, Shape{num_shared, inter_size, gate_up_groups, group_size}, {2});
        auto sh_scale_gate = op::v0::Constant::create(element::f16, Shape{num_shared, inter_size, gate_up_groups}, {0.02f});
        auto sh_zp_gate = op::v0::Constant::create(element::u4, Shape{num_shared, inter_size, gate_up_groups}, {0});
        auto sh_wei_up = op::v0::Constant::create(element::u4, Shape{num_shared, inter_size, gate_up_groups, group_size}, {2});
        auto sh_scale_up = op::v0::Constant::create(element::f16, Shape{num_shared, inter_size, gate_up_groups}, {0.02f});
        auto sh_zp_up = op::v0::Constant::create(element::u4, Shape{num_shared, inter_size, gate_up_groups}, {0});
        auto sh_wei_down = op::v0::Constant::create(element::u4, Shape{num_shared, hidden_size, down_groups, group_size}, {2});
        auto sh_scale_down = op::v0::Constant::create(element::f16, Shape{num_shared, hidden_size, down_groups}, {0.02f});
        auto sh_zp_down = op::v0::Constant::create(element::u4, Shape{num_shared, hidden_size, down_groups}, {0});
        auto sh_gate_gate_wei = op::v0::Constant::create(element::f16, Shape{hidden_size, num_shared}, {0.5f});

        auto routing_bias = op::v0::Constant::create(element::f16, Shape{1, num_experts}, {0.1f});
        auto routing_eps = op::v0::Constant::create(element::f16, Shape{1, 1}, {1e-6f});

        ov::op::internal::MOECompressed::Config config;
        config.expert_type = ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU;
        config.has_zp = true;
        config.hidden_size = hidden_size;
        config.inter_size = inter_size;
        config.num_expert = num_experts;
        config.num_shared_expert = num_shared;
        config.group_size = group_size;
        config.top_k = top_k;
        config.out_type = ov::element::f16;
        config.routing_type = ov::op::internal::MOECompressed::RoutingType::SIGMOID_BIAS;
        auto moe_3gemm_fused_compressed = std::make_shared<ov::intel_gpu::op::MOE3GemmFusedCompressed>(
            ov::OutputVector{hidden_states, routing_weights, wei_gate, scale_gate,   zp_gate,       wei_up,      scale_up,        zp_up,
                             wei_down,      scale_down,      zp_down,  routing_bias, routing_eps,   sh_wei_gate, sh_scale_gate,   sh_zp_gate,
                             sh_wei_up,     sh_scale_up,     sh_zp_up, sh_wei_down,  sh_scale_down, sh_zp_down,  sh_gate_gate_wei},
            config);

        model_ref = std::make_shared<ov::Model>(moe_3gemm_fused_compressed, ov::ParameterVector{hidden_states});
    }
}

TEST_F(TransformationTestsF, FuseMOE3GemmCompressedTest1) {
    // ── MoE shape parameters (larger experts, smaller inter_size, larger top_k) ──
    constexpr size_t batch = 4;
    constexpr size_t seq_len = 8;
    constexpr size_t tokens = batch * seq_len;
    constexpr size_t hidden_size = 2048;
    constexpr size_t inter_size = 512;
    constexpr size_t num_experts = 512;
    constexpr size_t top_k = 10;
    constexpr size_t group_size = 128;
    constexpr size_t gate_up_groups = hidden_size / group_size;  // gate/up grouped along K=hidden_size
    constexpr size_t down_groups = inter_size / group_size;      // down grouped along K=inter_size

    {
        auto hidden_states = std::make_shared<ov::op::v0::Parameter>(element::f16, Shape{batch, seq_len, hidden_size});
        auto flatten_shape = op::v0::Constant::create(element::i32, Shape{2}, {static_cast<int32_t>(tokens), static_cast<int32_t>(hidden_size)});
        auto hidden_states_reshape = std::make_shared<ov::op::v1::Reshape>(hidden_states, flatten_shape, false);
        auto routers = op::v0::Constant::create(element::f16, Shape{hidden_size, num_experts}, {0.2});
        // [tokens, num_experts]
        auto routing_weights = std::make_shared<ov::op::v0::MatMul>(hidden_states_reshape, routers);

        // Build post-GatherMatmul softmax routing (no ScatterElementsUpdate)
        auto [unsqueeze_moe, topk_indices] = build_softmax_routing_for_fuse_test(routing_weights, top_k);

        auto wei_gate = op::v0::Constant::create(element::u4, Shape{num_experts, inter_size, gate_up_groups, group_size}, {1});
        auto scale_gate = op::v0::Constant::create(element::f16, Shape{num_experts, inter_size, gate_up_groups}, {0.01f});
        auto zp_gate = op::v0::Constant::create(element::u4, Shape{num_experts, inter_size, gate_up_groups}, {0});
        auto wei_up = op::v0::Constant::create(element::u4, Shape{num_experts, inter_size, gate_up_groups, group_size}, {1});
        auto scale_up = op::v0::Constant::create(element::f16, Shape{num_experts, inter_size, gate_up_groups}, {0.01f});
        auto zp_up = op::v0::Constant::create(element::u4, Shape{num_experts, inter_size, gate_up_groups}, {0});
        auto wei_down = op::v0::Constant::create(element::u4, Shape{num_experts, hidden_size, down_groups, group_size}, {1});
        auto scale_down = op::v0::Constant::create(element::f16, Shape{num_experts, hidden_size, down_groups}, {0.01f});
        auto zp_down = op::v0::Constant::create(element::u4, Shape{num_experts, hidden_size, down_groups}, {0});

        ov::op::internal::MOECompressed::Config config;
        config.expert_type = ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU;
        config.has_zp = true;
        config.hidden_size = hidden_size;
        config.inter_size = inter_size;
        config.num_expert = num_experts;
        config.group_size = group_size;
        config.top_k = top_k;
        config.out_type = ov::element::f16;
        auto moe_compressed = std::make_shared<ov::op::internal::MOECompressed>(ov::OutputVector{hidden_states_reshape,
                                                                                                 unsqueeze_moe,
                                                                                                 topk_indices,
                                                                                                 wei_gate,
                                                                                                 scale_gate,
                                                                                                 zp_gate,
                                                                                                 wei_up,
                                                                                                 scale_up,
                                                                                                 zp_up,
                                                                                                 wei_down,
                                                                                                 scale_down,
                                                                                                 zp_down},
                                                                                config);
        model = std::make_shared<ov::Model>(moe_compressed, ov::ParameterVector{hidden_states});
        manager.register_pass<FuseMOE3GemmCompressed>();
    }
    {
        auto hidden_states = std::make_shared<ov::op::v0::Parameter>(element::f16, Shape{batch, seq_len, hidden_size});
        auto flatten_shape = op::v0::Constant::create(element::i32, Shape{2}, {static_cast<int32_t>(tokens), static_cast<int32_t>(hidden_size)});
        auto hidden_states_reshape = std::make_shared<ov::op::v1::Reshape>(hidden_states, flatten_shape, false);
        auto routers = op::v0::Constant::create(element::f16, Shape{hidden_size, num_experts}, {0.2});
        auto routing_weights = std::make_shared<ov::op::v0::MatMul>(hidden_states_reshape, routers);

        auto wei_gate = op::v0::Constant::create(element::u4, Shape{num_experts, inter_size, gate_up_groups, group_size}, {1});
        auto scale_gate = op::v0::Constant::create(element::f16, Shape{num_experts, inter_size, gate_up_groups}, {0.01f});
        auto zp_gate = op::v0::Constant::create(element::u4, Shape{num_experts, inter_size, gate_up_groups}, {0});
        auto wei_up = op::v0::Constant::create(element::u4, Shape{num_experts, inter_size, gate_up_groups, group_size}, {1});
        auto scale_up = op::v0::Constant::create(element::f16, Shape{num_experts, inter_size, gate_up_groups}, {0.01f});
        auto zp_up = op::v0::Constant::create(element::u4, Shape{num_experts, inter_size, gate_up_groups}, {0});
        auto wei_down = op::v0::Constant::create(element::u4, Shape{num_experts, hidden_size, down_groups, group_size}, {1});
        auto scale_down = op::v0::Constant::create(element::f16, Shape{num_experts, hidden_size, down_groups}, {0.01f});
        auto zp_down = op::v0::Constant::create(element::u4, Shape{num_experts, hidden_size, down_groups}, {0});

        ov::op::internal::MOECompressed::Config config;
        config.expert_type = ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU;
        config.has_zp = true;
        config.hidden_size = hidden_size;
        config.inter_size = inter_size;
        config.num_expert = num_experts;
        config.group_size = group_size;
        config.top_k = top_k;
        config.out_type = ov::element::f16;
        auto moe_3gemm_fused_compressed = std::make_shared<ov::intel_gpu::op::MOE3GemmFusedCompressed>(
            ov::OutputVector{hidden_states_reshape, routing_weights, wei_gate, scale_gate, zp_gate, wei_up, scale_up, zp_up, wei_down, scale_down, zp_down},
            config);

        model_ref = std::make_shared<ov::Model>(moe_3gemm_fused_compressed, ov::ParameterVector{hidden_states});
    }
}

// Qwen3.5-35B-A3B pattern
TEST_F(TransformationTestsF, FuseMOE3GemmCompressed_SoftmaxRouting_SliceAfterDivide) {
    // ── MoE shape parameters (shared between input model and reference model) ──
    constexpr size_t batch = 4;
    constexpr size_t seq_len = 8;
    constexpr size_t tokens = batch * seq_len;
    constexpr size_t hidden_size = 2048;
    constexpr size_t inter_size = 768;
    constexpr size_t num_experts = 128;
    constexpr size_t top_k = 8;
    constexpr size_t group_size = 128;
    constexpr size_t gate_up_groups = hidden_size / group_size;  // weight is grouped along K=hidden_size
    constexpr size_t down_groups = inter_size / group_size;      // weight is grouped along K=inter_size
    constexpr int64_t routing_axis = 1;                          // top-k axis on routing weights

    {
        // Hidden states: original 3D layout, flattened to 2D before the routing matmul.
        auto hidden_states = std::make_shared<ov::op::v0::Parameter>(element::f16, Shape{batch, seq_len, hidden_size});
        auto flatten_shape = op::v0::Constant::create(element::i32, Shape{2}, {static_cast<int32_t>(tokens), static_cast<int32_t>(hidden_size)});
        auto hidden_states_reshape = std::make_shared<ov::op::v1::Reshape>(hidden_states, flatten_shape, false);
        auto routers = op::v0::Constant::create(element::f16, Shape{hidden_size, num_experts}, {0.2});
        auto routing_weights = std::make_shared<ov::op::v0::MatMul>(hidden_states_reshape, routers);

        // Softmax routing with an extra Slice inserted between Divide and Transpose
        // (mirrors the Qwen3.5 graph: Softmax → TopK → ReduceSum → Divide → Slice → Transpose → Unsqueeze).
        auto softmax = std::make_shared<ov::op::v8::Softmax>(routing_weights, routing_axis);
        auto k = op::v0::Constant::create(element::i32, Shape{}, {static_cast<int32_t>(top_k)});
        auto topk_node = std::make_shared<ov::op::v11::TopK>(softmax, k, routing_axis, ov::op::v11::TopK::Mode::MAX, ov::op::v11::TopK::SortType::SORT_VALUES);
        auto reduce_axis = op::v0::Constant::create(element::i64, Shape{1}, {routing_axis});
        auto reduce_sum = std::make_shared<ov::op::v1::ReduceSum>(topk_node->output(0), reduce_axis, true);
        auto norm = std::make_shared<ov::op::v1::Divide>(topk_node->output(0), reduce_sum);

        auto slice_start = op::v0::Constant::create(element::i64, Shape{1}, {0});
        auto topk_idx_convert = std::make_shared<ov::op::v0::Convert>(topk_node->output(1), element::i32);
        auto topk_idx_shape = std::make_shared<ov::op::v3::ShapeOf>(topk_idx_convert);
        auto gather_idx = op::v0::Constant::create(element::i64, Shape{1}, {routing_axis});
        auto gather_axis = op::v0::Constant::create(element::i64, Shape{}, {0});
        auto slice_stop = std::make_shared<ov::op::v8::Gather>(topk_idx_shape, gather_idx, gather_axis);
        auto slice_step = op::v0::Constant::create(element::i64, Shape{1}, {1});
        auto slice_axis = op::v0::Constant::create(element::i64, Shape{1}, {routing_axis});
        auto slice = std::make_shared<ov::op::v8::Slice>(norm, slice_start, slice_stop, slice_step, slice_axis);

        auto transpose_order = op::v0::Constant::create(element::i64, Shape{2}, {1, 0});
        auto transpose = std::make_shared<ov::op::v1::Transpose>(slice, transpose_order);
        auto unsqueeze_const = op::v0::Constant::create(element::i64, Shape{1}, {-1});
        auto unsqueeze_moe = std::make_shared<ov::op::v0::Unsqueeze>(transpose, unsqueeze_const);

        // Compressed weights: 4-D weight = {experts, N, num_groups, group_size};
        // 3-D scale/zp = {experts, N, num_groups}. The K axis is split into groups of
        // `group_size`, so num_groups * group_size = K (gate/up: K=hidden_size, down: K=inter_size).
        auto wei_gate = op::v0::Constant::create(element::u4, Shape{num_experts, inter_size, gate_up_groups, group_size}, {1});
        auto scale_gate = op::v0::Constant::create(element::f16, Shape{num_experts, inter_size, gate_up_groups}, {0.01f});
        auto zp_gate = op::v0::Constant::create(element::u4, Shape{num_experts, inter_size, gate_up_groups}, {0});
        auto wei_up = op::v0::Constant::create(element::u4, Shape{num_experts, inter_size, gate_up_groups, group_size}, {1});
        auto scale_up = op::v0::Constant::create(element::f16, Shape{num_experts, inter_size, gate_up_groups}, {0.01f});
        auto zp_up = op::v0::Constant::create(element::u4, Shape{num_experts, inter_size, gate_up_groups}, {0});
        auto wei_down = op::v0::Constant::create(element::u4, Shape{num_experts, hidden_size, down_groups, group_size}, {1});
        auto scale_down = op::v0::Constant::create(element::f16, Shape{num_experts, hidden_size, down_groups}, {0.01f});
        auto zp_down = op::v0::Constant::create(element::u4, Shape{num_experts, hidden_size, down_groups}, {0});

        ov::op::internal::MOECompressed::Config config;
        config.expert_type = ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU;
        config.has_zp = true;
        config.hidden_size = hidden_size;
        config.inter_size = inter_size;
        config.num_expert = num_experts;
        config.group_size = group_size;
        config.top_k = top_k;
        config.out_type = ov::element::f16;

        // Reuse `topk_idx_convert` as MOECompressed's indices input — that mirrors the real
        // graph's `MOECompressed(..., Convert, ...)` and keeps the Convert with two consumers
        // (ShapeOf and MOECompressed), neither pattern-constrained on consumers count.
        auto moe_compressed = std::make_shared<ov::op::internal::MOECompressed>(ov::OutputVector{hidden_states_reshape,
                                                                                                 unsqueeze_moe,
                                                                                                 topk_idx_convert,
                                                                                                 wei_gate,
                                                                                                 scale_gate,
                                                                                                 zp_gate,
                                                                                                 wei_up,
                                                                                                 scale_up,
                                                                                                 zp_up,
                                                                                                 wei_down,
                                                                                                 scale_down,
                                                                                                 zp_down},
                                                                                config);
        model = std::make_shared<ov::Model>(moe_compressed, ov::ParameterVector{hidden_states});
    }
    manager.register_pass<FuseMOE3GemmCompressed>();
    {
        auto hidden_states_ref = std::make_shared<ov::op::v0::Parameter>(element::f16, Shape{batch, seq_len, hidden_size});
        auto flatten_shape_ref = op::v0::Constant::create(element::i32, Shape{2}, {static_cast<int32_t>(tokens), static_cast<int32_t>(hidden_size)});
        auto hidden_states_reshape_ref = std::make_shared<ov::op::v1::Reshape>(hidden_states_ref, flatten_shape_ref, false);
        auto routers_ref = op::v0::Constant::create(element::f16, Shape{hidden_size, num_experts}, {0.2});
        auto routing_weights_ref = std::make_shared<ov::op::v0::MatMul>(hidden_states_reshape_ref, routers_ref);

        auto wei_gate_ref = op::v0::Constant::create(element::u4, Shape{num_experts, inter_size, gate_up_groups, group_size}, {1});
        auto scale_gate_ref = op::v0::Constant::create(element::f16, Shape{num_experts, inter_size, gate_up_groups}, {0.01f});
        auto zp_gate_ref = op::v0::Constant::create(element::u4, Shape{num_experts, inter_size, gate_up_groups}, {0});
        auto wei_up_ref = op::v0::Constant::create(element::u4, Shape{num_experts, inter_size, gate_up_groups, group_size}, {1});
        auto scale_up_ref = op::v0::Constant::create(element::f16, Shape{num_experts, inter_size, gate_up_groups}, {0.01f});
        auto zp_up_ref = op::v0::Constant::create(element::u4, Shape{num_experts, inter_size, gate_up_groups}, {0});
        auto wei_down_ref = op::v0::Constant::create(element::u4, Shape{num_experts, hidden_size, down_groups, group_size}, {1});
        auto scale_down_ref = op::v0::Constant::create(element::f16, Shape{num_experts, hidden_size, down_groups}, {0.01f});
        auto zp_down_ref = op::v0::Constant::create(element::u4, Shape{num_experts, hidden_size, down_groups}, {0});

        ov::op::internal::MOECompressed::Config config_ref;
        config_ref.expert_type = ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU;
        config_ref.has_zp = true;
        config_ref.hidden_size = hidden_size;
        config_ref.inter_size = inter_size;
        config_ref.num_expert = num_experts;
        config_ref.group_size = group_size;
        config_ref.top_k = top_k;
        config_ref.out_type = ov::element::f16;
        config_ref.routing_type = ov::op::internal::MOECompressed::RoutingType::SOFTMAX;

        ov::OutputVector args{hidden_states_reshape_ref,
                              routing_weights_ref,
                              wei_gate_ref,
                              scale_gate_ref,
                              zp_gate_ref,
                              wei_up_ref,
                              scale_up_ref,
                              zp_up_ref,
                              wei_down_ref,
                              scale_down_ref,
                              zp_down_ref};
        auto fused = std::make_shared<ov::intel_gpu::op::MOE3GemmFusedCompressed>(args, config_ref);
        model_ref = std::make_shared<ov::Model>(fused, ov::ParameterVector{hidden_states_ref});
    }
}

}  // namespace intel_gpu
}  // namespace test
}  // namespace ov
