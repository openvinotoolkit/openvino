// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <openvino/core/model.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/clamp.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/gelu.hpp>
#include <openvino/op/matmul.hpp>
#include <openvino/op/minimum.hpp>
#include <openvino/op/moe.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/reduce_sum.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/slice.hpp>
#include <openvino/op/swish.hpp>
#include <openvino/op/tile.hpp>
#include <openvino/op/topk.hpp>
#include <openvino/pass/manager.hpp>
#include <openvino/pass/serialize.hpp>
#include <openvino/pass/visualize_tree.hpp>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "common_test_utils/subgraph_builders/weights_decompression_builders.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "ov_ops/gather_matmul.hpp"
#include "ov_ops/gather_matmul_compressed.hpp"
#include "ov_ops/moe_compressed.hpp"
#include "transformations/common_optimizations/convert_tiled_moe_block_to_gather_matmuls.hpp"
#include "transformations/common_optimizations/moe_op_fusion.hpp"

using GatherMatmul = ov::op::internal::GatherMatmul;

// ============================================================================
// IR model builders (original MOE pattern before any transformation)
// ============================================================================

inline std::shared_ptr<ov::Model> build_2gemm_moe_pattern_model() {
    using namespace ov;

    const size_t batch = 2;
    const Dimension in_dim = Dimension::dynamic();
    const size_t hidden_size = 2048;
    const size_t intermediate_size = 4096;
    const size_t topk = 2;
    const size_t number_of_experts = 3;
    const size_t fusion_factor = 2;
    const auto expert_alpha = 1.702f;
    const auto expert_beta = 7.0f;

    auto input_shape = PartialShape{batch, in_dim, hidden_size};
    auto input = std::make_shared<op::v0::Parameter>(element::f32, input_shape);
    auto experts_reshape = std::make_shared<op::v1::Reshape>(
        input,
        op::v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{-1, hidden_size}),
        false);

    auto tile = std::make_shared<op::v0::Tile>(
        experts_reshape,
        op::v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{number_of_experts, 1}));
    auto after_tile_reshape = std::make_shared<op::v1::Reshape>(
        tile,
        op::v0::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{number_of_experts, batch, hidden_size}),
        false);

    auto gate_up_matmul = std::make_shared<op::v0::MatMul>(
        after_tile_reshape,
        op::v0::Constant::create(element::f32,
                                 Shape{number_of_experts, intermediate_size * fusion_factor, hidden_size},
                                 {1.0f}),
        false,
        true);
    auto gate_up_add = std::make_shared<op::v1::Add>(
        gate_up_matmul,
        op::v0::Constant::create(element::f32, Shape{number_of_experts, 1, intermediate_size * fusion_factor}, {0.0f}));

    auto slice1 = std::make_shared<op::v8::Slice>(
        gate_up_add,
        op::v0::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{0, 0, 0}),
        op::v0::Constant::create(element::i64,
                                 Shape{3},
                                 std::vector<int64_t>{number_of_experts, batch, intermediate_size * 2}),
        op::v0::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{1, 1, 2}),
        op::v0::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{0, 1, 2}));
    auto clamp = std::make_shared<op::v0::Clamp>(slice1, -expert_beta, expert_beta);
    auto add1 = std::make_shared<op::v1::Add>(clamp, op::v0::Constant::create(element::f32, Shape{1}, {1.0f}));

    auto slice2 = std::make_shared<op::v8::Slice>(
        gate_up_add,
        op::v0::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{0, 1, 0}),
        op::v0::Constant::create(element::i64,
                                 Shape{3},
                                 std::vector<int64_t>{number_of_experts, batch, intermediate_size * 2}),
        op::v0::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{1, 1, 2}),
        op::v0::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{0, 1, 2}));
    auto minimum1 =
        std::make_shared<op::v1::Minimum>(slice2, op::v0::Constant::create(element::f32, Shape{1}, {10.0f}));
    auto swish_beta = op::v0::Constant::create(element::f32, Shape{}, std::vector<float>{expert_alpha});
    auto swish = std::make_shared<op::v4::Swish>(minimum1, swish_beta);

    auto multiply2 = std::make_shared<op::v1::Multiply>(add1, swish);

    auto down_proj_matmul = std::make_shared<op::v0::MatMul>(
        multiply2,
        op::v0::Constant::create(element::f32, Shape{number_of_experts, hidden_size, intermediate_size}, {1.0f}),
        false,
        true);

    auto down_proj_add = std::make_shared<op::v1::Add>(
        down_proj_matmul,
        op::v0::Constant::create(element::f32, Shape{number_of_experts, 1, hidden_size}, {1.0f}));

    auto end_reshape = std::make_shared<op::v1::Reshape>(
        down_proj_add,
        op::v0::Constant::create(element::i64,
                                 Shape{4},
                                 std::vector<int64_t>{number_of_experts, batch, -1, hidden_size}),
        false);

    // Router subgraph
    auto reshape_2nd_consumer_router_matmul = std::make_shared<op::v0::MatMul>(
        experts_reshape,
        op::v0::Constant::create(element::f32, Shape{number_of_experts, hidden_size}, {1.0f}),
        false,
        true);

    auto router_bias =
        std::make_shared<op::v1::Add>(reshape_2nd_consumer_router_matmul,
                                      op::v0::Constant::create(element::f32, Shape{1, number_of_experts}, {1.0f}));

    auto router_topk_values_and_indices =
        std::make_shared<op::v11::TopK>(router_bias,
                                        op::v0::Constant::create(element::i64, Shape{}, {topk}),
                                        -1,
                                        op::v11::TopK::Mode::MAX,
                                        op::v11::TopK::SortType::SORT_VALUES,
                                        element::i64);

    auto router_topk_values = router_topk_values_and_indices->output(0);
    auto router_topk_indices = router_topk_values_and_indices->output(1);

    auto scatter_elements_update = std::make_shared<op::v12::ScatterElementsUpdate>(
        router_topk_values,
        router_topk_indices,
        op::v0::Constant::create(element::f32, Shape{batch, topk}, {0}),
        op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{1}));
    auto router_transpose = std::make_shared<op::v1::Transpose>(
        scatter_elements_update,
        op::v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{1, 0}));
    auto router_reshape = std::make_shared<op::v1::Reshape>(
        router_transpose,
        op::v0::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{number_of_experts, batch, -1}),
        true);
    auto unsqueeze_routing_weights =
        std::make_shared<op::v0::Unsqueeze>(router_reshape,
                                            op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{-1}));

    auto mul3 = std::make_shared<op::v1::Multiply>(end_reshape, unsqueeze_routing_weights);

    auto reduce_sum =
        std::make_shared<op::v1::ReduceSum>(mul3,
                                            op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{0}),
                                            false);

    return std::make_shared<ov::Model>(ov::OutputVector{reduce_sum}, ov::ParameterVector{input});
}

inline std::shared_ptr<ov::Model> build_3gemm_moe_pattern_model() {
    using namespace ov;

    const size_t batch = 2;
    const Dimension in_dim = Dimension::dynamic();
    const size_t hidden_size = 2048;
    const size_t intermediate_size = 4096;
    const size_t number_of_experts = 3;
    const size_t topk = 2;

    auto input_shape = PartialShape{batch, in_dim, hidden_size};
    auto input = std::make_shared<op::v0::Parameter>(element::f32, input_shape);
    auto experts_reshape = std::make_shared<op::v1::Reshape>(
        input,
        op::v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{-1, hidden_size}),
        false);

    auto tile = std::make_shared<op::v0::Tile>(
        experts_reshape,
        op::v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{number_of_experts, 1}));
    auto after_tile_reshape = std::make_shared<op::v1::Reshape>(
        tile,
        op::v0::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{number_of_experts, batch, hidden_size}),
        false);

    // First GEMM (gate)
    auto gate_matmul = std::make_shared<op::v0::MatMul>(
        after_tile_reshape,
        op::v0::Constant::create(element::f32, Shape{number_of_experts, intermediate_size, hidden_size}, {1.0f}),
        false,
        true);

    auto swish = std::make_shared<op::v4::Swish>(gate_matmul);

    // Second GEMM (up)
    auto up_matmul = std::make_shared<op::v0::MatMul>(
        after_tile_reshape,
        op::v0::Constant::create(element::f32, Shape{number_of_experts, intermediate_size, hidden_size}, {1.0f}),
        false,
        true);

    auto swiglu = std::make_shared<op::v1::Multiply>(swish, up_matmul);

    // Third GEMM (down)
    auto down_matmul = std::make_shared<op::v0::MatMul>(
        swiglu,
        op::v0::Constant::create(element::f32, Shape{number_of_experts, hidden_size, intermediate_size}, {1.0f}),
        false,
        true);

    auto experts_out_reshape = std::make_shared<op::v1::Reshape>(
        down_matmul,
        op::v0::Constant::create(element::i64,
                                 Shape{4},
                                 std::vector<int64_t>{number_of_experts, batch, -1, hidden_size}),
        false);

    // Router subgraph
    auto router_matmul = std::make_shared<op::v0::MatMul>(
        experts_reshape,
        op::v0::Constant::create(element::f32, Shape{number_of_experts, hidden_size}, {1.0f}),
        false,
        true);

    auto router_topk_values_and_indices =
        std::make_shared<op::v11::TopK>(router_matmul,
                                        op::v0::Constant::create(element::i64, Shape{}, {topk}),
                                        -1,
                                        op::v11::TopK::Mode::MAX,
                                        op::v11::TopK::SortType::SORT_VALUES,
                                        element::i64);

    auto router_topk_values = router_topk_values_and_indices->output(0);
    auto router_topk_indices = router_topk_values_and_indices->output(1);

    auto scatter_elements_update = std::make_shared<op::v12::ScatterElementsUpdate>(
        router_topk_values,
        router_topk_indices,
        op::v0::Constant::create(element::f32, Shape{batch, topk}, {0}),
        op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{1}));
    auto router_transpose = std::make_shared<op::v1::Transpose>(
        scatter_elements_update,
        op::v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{1, 0}));
    auto router_reshape = std::make_shared<op::v1::Reshape>(
        router_transpose,
        op::v0::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{number_of_experts, batch, -1}),
        true);
    auto unsqueeze_routing_weights =
        std::make_shared<op::v0::Unsqueeze>(router_reshape,
                                            op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{-1}));

    auto mul3 = std::make_shared<op::v1::Multiply>(experts_out_reshape, unsqueeze_routing_weights);

    auto reduce_sum =
        std::make_shared<op::v1::ReduceSum>(mul3,
                                            op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{0}),
                                            false);

    return std::make_shared<ov::Model>(ov::OutputVector{reduce_sum}, ov::ParameterVector{input});
}

// ============================================================================
// Post-BGM model builders (3 BGMs + compact routing + ReduceSum + Reshape)
// ============================================================================

// Compressed expert weights: wei_type weights, f16 scale [E, N, K / group_size] and zp
struct MoeCompression {
    static constexpr size_t group_size = 128;
    ov::element::Type wei_type;
    ov::test::utils::DecompressionType decompression_subtract_type;
    ov::element::Type zp_type = ov::element::dynamic;
};

// Returns {weights} for plain experts, {weights, scale, zp} for compressed ones
inline ov::OutputVector make_expert_weights(size_t number_of_experts,
                                            size_t n,
                                            size_t k,
                                            const std::optional<MoeCompression>& compression) {
    using namespace ov;
    if (!compression) {
        return {op::v0::Constant::create(element::f32, Shape{number_of_experts, n, k}, {1.0f})};
    }
    using ov::test::utils::DecompressionType;
    const Shape scale_shape{number_of_experts, n, k / MoeCompression::group_size};
    const auto zp_decompression = compression->decompression_subtract_type;
    const auto zp =
        zp_decompression == DecompressionType::empty
            ? std::make_shared<op::v0::Constant>(element::dynamic, Shape{0})
            : op::v0::Constant::create(compression->zp_type,
                                       zp_decompression == DecompressionType::scalar ? Shape{} : scale_shape,
                                       {4});
    return {op::v0::Constant::create(compression->wei_type, Shape{number_of_experts, n, k}, {1}),
            op::v0::Constant::create(element::f16, scale_shape, {1.0f}),
            zp};
}

inline std::shared_ptr<ov::Model> build_3gemm_bgm_model(
    ov::op::internal::MOE::Activation_type activation_type = ov::op::internal::MOE::Activation_type::SWIGLU,
    bool has_batch_dim = true,
    bool restore_output_shape = false,
    const std::optional<MoeCompression>& compression = std::nullopt) {
    using namespace ov;

    const size_t batch = 2;
    const Dimension in_dim = Dimension::dynamic();
    const size_t hidden_size = 2048;
    const size_t intermediate_size = 4096;
    const size_t number_of_experts = 3;
    const size_t topk = 2;

    auto input_shape = has_batch_dim ? PartialShape{batch, in_dim, hidden_size} : PartialShape{in_dim, 1, hidden_size};
    auto input = std::make_shared<op::v0::Parameter>(element::f32, input_shape);
    auto experts_reshape = std::make_shared<op::v1::Reshape>(
        input,
        op::v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{-1, static_cast<int64_t>(hidden_size)}),
        false);

    // Unsqueeze to add experts dimension: [1, batch*seq, hidden]
    auto unsqueeze =
        std::make_shared<op::v0::Unsqueeze>(experts_reshape, op::v0::Constant::create(element::i32, Shape{}, {0}));

    // Router subgraph to produce topk_indices and chosen_experts
    auto router_matmul = std::make_shared<op::v0::MatMul>(
        experts_reshape,
        op::v0::Constant::create(element::f32, Shape{number_of_experts, hidden_size}, {1.0f}),
        false,
        true);
    auto router_topk = std::make_shared<op::v11::TopK>(router_matmul,
                                                       op::v0::Constant::create(element::i64, Shape{}, {topk}),
                                                       -1,
                                                       op::v11::TopK::Mode::MAX,
                                                       op::v11::TopK::SortType::SORT_VALUES,
                                                       element::i64);
    auto topk_indices = router_topk->output(1);    // [batch*seq, topk]
    auto chosen_experts = router_topk->output(0);  // [batch*seq, topk] (values used as routing weights)

    auto make_bgm = [&](const Output<Node>& a, size_t n, size_t k) -> std::shared_ptr<Node> {
        const auto w = make_expert_weights(number_of_experts, n, k, compression);
        if (!compression) {
            return std::make_shared<GatherMatmul>(a, w[0], topk_indices);
        }
        auto bias = std::make_shared<op::v0::Constant>(element::dynamic, Shape{0});
        return std::make_shared<ov::op::internal::GatherMatmulCompressed>(a, w[0], topk_indices, bias, w[1], w[2]);
    };

    // 3 BGMs
    auto bgm_gate = make_bgm(unsqueeze, intermediate_size, hidden_size);
    std::shared_ptr<ov::Node> gate_act;
    if (activation_type == ov::op::internal::MOE::Activation_type::GEGLU_TANH) {
        gate_act = std::make_shared<op::v7::Gelu>(bgm_gate, ov::op::GeluApproximationMode::TANH);
    } else if (activation_type == ov::op::internal::MOE::Activation_type::GEGLU_ERF) {
        gate_act = std::make_shared<op::v7::Gelu>(bgm_gate, ov::op::GeluApproximationMode::ERF);
    } else {
        gate_act = std::make_shared<op::v4::Swish>(bgm_gate);
    }
    auto bgm_up = make_bgm(unsqueeze, intermediate_size, hidden_size);
    auto swiglu = std::make_shared<op::v1::Multiply>(gate_act, bgm_up);
    auto bgm_down = make_bgm(swiglu, hidden_size, intermediate_size);

    // Compact routing: chosen_experts → Transpose({1,0}) → Unsqueeze(-1)
    auto router_transpose = std::make_shared<op::v1::Transpose>(
        chosen_experts,
        op::v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{1, 0}));
    auto router_unsqueeze =
        std::make_shared<op::v0::Unsqueeze>(router_transpose, op::v0::Constant::create(element::i32, Shape{}, {-1}));

    // Final: Multiply → ReduceSum → Reshape
    auto final_mul = std::make_shared<op::v1::Multiply>(bgm_down, router_unsqueeze);
    auto reduce_sum =
        std::make_shared<op::v1::ReduceSum>(final_mul,
                                            op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{0}),
                                            false);

    const std::vector<int64_t> output_shape =
        restore_output_shape ? std::vector<int64_t>{-1, static_cast<int64_t>(batch), static_cast<int64_t>(hidden_size)}
        : has_batch_dim      ? std::vector<int64_t>{static_cast<int64_t>(batch), -1, static_cast<int64_t>(hidden_size)}
                             : std::vector<int64_t>{-1, static_cast<int64_t>(hidden_size)};
    auto end_reshape = std::make_shared<op::v1::Reshape>(
        reduce_sum,
        op::v0::Constant::create(element::i64, Shape{output_shape.size()}, output_shape),
        true);

    return std::make_shared<ov::Model>(ov::OutputVector{end_reshape}, ov::ParameterVector{input});
}

inline std::shared_ptr<ov::Model> build_3gemm_bgm_to_moe_reference_model(
    ov::op::internal::MOE::Activation_type activation_type = ov::op::internal::MOE::Activation_type::SWIGLU,
    bool has_batch_dim = true,
    bool restore_output_shape = false,
    const std::optional<MoeCompression>& compression = std::nullopt) {
    using namespace ov;

    const size_t batch = 2;
    const Dimension in_dim = Dimension::dynamic();
    const size_t hidden_size = 2048;
    const size_t intermediate_size = 4096;
    const size_t number_of_experts = 3;
    const size_t topk = 2;

    auto input_shape = has_batch_dim ? PartialShape{batch, in_dim, hidden_size} : PartialShape{in_dim, 1, hidden_size};
    auto input = std::make_shared<op::v0::Parameter>(element::f32, input_shape);

    // Router subgraph (not fused, remains in the graph)
    auto experts_reshape = std::make_shared<op::v1::Reshape>(
        input,
        op::v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{-1, static_cast<int64_t>(hidden_size)}),
        false);

    auto router_matmul = std::make_shared<op::v0::MatMul>(
        experts_reshape,
        op::v0::Constant::create(element::f32, Shape{number_of_experts, hidden_size}, {1.0f}),
        false,
        true);
    auto router_topk = std::make_shared<op::v11::TopK>(router_matmul,
                                                       op::v0::Constant::create(element::i64, Shape{}, {topk}),
                                                       -1,
                                                       op::v11::TopK::Mode::MAX,
                                                       op::v11::TopK::SortType::SORT_VALUES,
                                                       element::i64);
    auto topk_indices = router_topk->output(1);
    auto routing = router_topk->output(0);

    // MOE op with compact routing
    auto hidden_states = input->output(0);
    ov::OutputVector moe_inputs = {hidden_states, routing, topk_indices};
    for (auto [n, k] : {std::pair{intermediate_size, hidden_size},
                        std::pair{intermediate_size, hidden_size},
                        std::pair{hidden_size, intermediate_size}}) {
        const auto w = make_expert_weights(number_of_experts, n, k, compression);
        moe_inputs.insert(moe_inputs.end(), w.begin(), w.end());
    }

    ov::Output<ov::Node> output;
    if (compression) {
        using ov::op::internal::MOECompressed;
        MOECompressed::Config config{
            {ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU, 0.0f, 1.0f, 0, activation_type},
            hidden_size,
            intermediate_size,
            number_of_experts,
            0,  // num_shared_expert
            topk,
            MoeCompression::group_size,
            has_batch_dim,
            moe_inputs[5].get_element_type() != element::dynamic,  // has_zp
            element::f16,
        };
        auto moe = std::make_shared<MOECompressed>(moe_inputs, config);
        output = std::make_shared<op::v0::Convert>(moe, input->get_element_type());
    } else {
        ov::op::internal::MOE::Config config;
        config.expert_type = ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU;
        config.activation_type = activation_type;
        output = std::make_shared<ov::op::internal::MOE>(moe_inputs, config);
    }

    if (!has_batch_dim || restore_output_shape) {
        const std::vector<int64_t> output_shape =
            restore_output_shape
                ? std::vector<int64_t>{-1, static_cast<int64_t>(batch), static_cast<int64_t>(hidden_size)}
                : std::vector<int64_t>{-1, static_cast<int64_t>(hidden_size)};
        output = std::make_shared<op::v1::Reshape>(
            output,
            op::v0::Constant::create(element::i64, Shape{output_shape.size()}, output_shape),
            true);
    }
    return std::make_shared<ov::Model>(ov::OutputVector{output}, ov::ParameterVector{input});
}

inline std::shared_ptr<ov::Model> build_2gemm_bgm_model(bool has_batch_dim = true, bool restore_output_shape = false) {
    using namespace ov;

    const size_t batch = 2;
    const Dimension in_dim = Dimension::dynamic();
    const size_t hidden_size = 2048;
    const size_t intermediate_size = 4096;
    const size_t topk = 2;
    const size_t number_of_experts = 3;
    const size_t fusion_factor = 2;
    const auto expert_alpha = 1.702f;
    const auto expert_beta = 7.0f;

    auto input_shape = has_batch_dim ? PartialShape{batch, in_dim, hidden_size} : PartialShape{in_dim, 1, hidden_size};
    auto input = std::make_shared<op::v0::Parameter>(element::f32, input_shape);
    auto experts_reshape = std::make_shared<op::v1::Reshape>(
        input,
        op::v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{-1, static_cast<int64_t>(hidden_size)}),
        false);

    auto unsqueeze =
        std::make_shared<op::v0::Unsqueeze>(experts_reshape, op::v0::Constant::create(element::i32, Shape{}, {0}));

    // Router
    auto router_matmul = std::make_shared<op::v0::MatMul>(
        experts_reshape,
        op::v0::Constant::create(element::f32, Shape{number_of_experts, hidden_size}, {1.0f}),
        false,
        true);
    auto router_bias =
        std::make_shared<op::v1::Add>(router_matmul,
                                      op::v0::Constant::create(element::f32, Shape{1, number_of_experts}, {1.0f}));
    auto router_topk = std::make_shared<op::v11::TopK>(router_bias,
                                                       op::v0::Constant::create(element::i64, Shape{}, {topk}),
                                                       -1,
                                                       op::v11::TopK::Mode::MAX,
                                                       op::v11::TopK::SortType::SORT_VALUES,
                                                       element::i64);
    auto topk_indices = router_topk->output(1);
    auto chosen_experts = router_topk->output(0);

    // Gate/up weights and bias
    auto gate_up_w = op::v0::Constant::create(element::f32,
                                              Shape{number_of_experts, intermediate_size * fusion_factor, hidden_size},
                                              {1.0f});
    auto gate_up_bias =
        op::v0::Constant::create(element::f32, Shape{number_of_experts, 1, intermediate_size * fusion_factor}, {0.0f});

    // BGM gate_up (4 inputs: data, weight, indices, bias)
    auto bgm_gate_up = std::make_shared<GatherMatmul>(unsqueeze, gate_up_w, topk_indices, gate_up_bias);

    // Activation subgraph (same as in the original 2GEMM pattern)
    auto slice1 = std::make_shared<op::v8::Slice>(
        bgm_gate_up,
        op::v0::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{0, 0, 0}),
        op::v0::Constant::create(element::i64,
                                 Shape{3},
                                 std::vector<int64_t>{static_cast<int64_t>(topk),
                                                      static_cast<int64_t>(batch),
                                                      static_cast<int64_t>(intermediate_size * 2)}),
        op::v0::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{1, 1, 2}),
        op::v0::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{0, 1, 2}));
    auto clamp = std::make_shared<op::v0::Clamp>(slice1, -expert_beta, expert_beta);
    auto add1 = std::make_shared<op::v1::Add>(clamp, op::v0::Constant::create(element::f32, Shape{1}, {1.0f}));

    auto slice2 = std::make_shared<op::v8::Slice>(
        bgm_gate_up,
        op::v0::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{0, 1, 0}),
        op::v0::Constant::create(element::i64,
                                 Shape{3},
                                 std::vector<int64_t>{static_cast<int64_t>(topk),
                                                      static_cast<int64_t>(batch),
                                                      static_cast<int64_t>(intermediate_size * 2)}),
        op::v0::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{1, 1, 2}),
        op::v0::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{0, 1, 2}));
    auto minimum1 =
        std::make_shared<op::v1::Minimum>(slice2, op::v0::Constant::create(element::f32, Shape{1}, {10.0f}));
    auto swish_beta_const = op::v0::Constant::create(element::f32, Shape{}, std::vector<float>{expert_alpha});
    auto swish = std::make_shared<op::v4::Swish>(minimum1, swish_beta_const);
    auto multiply2 = std::make_shared<op::v1::Multiply>(add1, swish);

    // Down proj
    auto down_w =
        op::v0::Constant::create(element::f32, Shape{number_of_experts, hidden_size, intermediate_size}, {1.0f});
    auto down_bias = op::v0::Constant::create(element::f32, Shape{number_of_experts, 1, hidden_size}, {1.0f});
    auto bgm_down = std::make_shared<GatherMatmul>(multiply2, down_w, topk_indices, down_bias);

    // Compact routing
    auto router_transpose = std::make_shared<op::v1::Transpose>(
        chosen_experts,
        op::v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{1, 0}));
    auto router_unsqueeze =
        std::make_shared<op::v0::Unsqueeze>(router_transpose, op::v0::Constant::create(element::i32, Shape{}, {-1}));

    auto final_mul = std::make_shared<op::v1::Multiply>(bgm_down, router_unsqueeze);
    auto reduce_sum =
        std::make_shared<op::v1::ReduceSum>(final_mul,
                                            op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{0}),
                                            false);
    const std::vector<int64_t> output_shape =
        restore_output_shape ? std::vector<int64_t>{-1, static_cast<int64_t>(batch), static_cast<int64_t>(hidden_size)}
        : has_batch_dim      ? std::vector<int64_t>{static_cast<int64_t>(batch), -1, static_cast<int64_t>(hidden_size)}
                             : std::vector<int64_t>{-1, static_cast<int64_t>(hidden_size)};
    auto end_reshape = std::make_shared<op::v1::Reshape>(
        reduce_sum,
        op::v0::Constant::create(element::i64, Shape{output_shape.size()}, output_shape),
        true);

    return std::make_shared<ov::Model>(ov::OutputVector{end_reshape}, ov::ParameterVector{input});
}

inline std::shared_ptr<ov::Model> build_2gemm_bgm_to_moe_reference_model(bool has_batch_dim = true,
                                                                         bool restore_output_shape = false) {
    using namespace ov;

    const size_t batch = 2;
    const Dimension in_dim = Dimension::dynamic();
    const size_t hidden_size = 2048;
    const size_t intermediate_size = 4096;
    const size_t topk = 2;
    const size_t number_of_experts = 3;
    const size_t fusion_factor = 2;
    const auto expert_alpha = 1.702f;
    const auto expert_beta = 7.0f;

    auto input_shape = has_batch_dim ? PartialShape{batch, in_dim, hidden_size} : PartialShape{in_dim, 1, hidden_size};
    auto input = std::make_shared<op::v0::Parameter>(element::f32, input_shape);

    auto experts_reshape = std::make_shared<op::v1::Reshape>(
        input,
        op::v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{-1, static_cast<int64_t>(hidden_size)}),
        false);

    // Router (stays in graph)
    auto router_matmul = std::make_shared<op::v0::MatMul>(
        experts_reshape,
        op::v0::Constant::create(element::f32, Shape{number_of_experts, hidden_size}, {1.0f}),
        false,
        true);
    auto router_bias =
        std::make_shared<op::v1::Add>(router_matmul,
                                      op::v0::Constant::create(element::f32, Shape{1, number_of_experts}, {1.0f}));
    auto router_topk = std::make_shared<op::v11::TopK>(router_bias,
                                                       op::v0::Constant::create(element::i64, Shape{}, {topk}),
                                                       -1,
                                                       op::v11::TopK::Mode::MAX,
                                                       op::v11::TopK::SortType::SORT_VALUES,
                                                       element::i64);
    auto topk_indices = router_topk->output(1);
    auto chosen_experts = router_topk->output(0);

    // Convert2GatherMatmulMoeBlockToMoeOp bypasses Transpose+Unsqueeze (tokens-major).
    auto routing = chosen_experts;

    // Weights
    auto gate_up_w = op::v0::Constant::create(element::f32,
                                              Shape{number_of_experts, intermediate_size * fusion_factor, hidden_size},
                                              {1.0f});
    auto gate_up_bias =
        op::v0::Constant::create(element::f32, Shape{number_of_experts, 1, intermediate_size * fusion_factor}, {0.0f});
    auto down_w =
        op::v0::Constant::create(element::f32, Shape{number_of_experts, hidden_size, intermediate_size}, {1.0f});
    auto down_bias = op::v0::Constant::create(element::f32, Shape{number_of_experts, 1, hidden_size}, {1.0f});

    auto hidden_states = input->output(0);
    ov::OutputVector moe_inputs = {hidden_states, routing, topk_indices, gate_up_w, gate_up_bias, down_w, down_bias};

    ov::op::internal::MOE::Config config;
    config.expert_type = ov::op::internal::MOE::Expert_type::GEMM2_BIAS_SWIGLU_CLAMP;
    config.expert_alpha = expert_alpha;
    config.expert_beta = expert_beta;

    auto moe = std::make_shared<ov::op::internal::MOE>(moe_inputs, config);
    ov::Output<ov::Node> output = moe;
    if (!has_batch_dim || restore_output_shape) {
        const std::vector<int64_t> output_shape =
            restore_output_shape
                ? std::vector<int64_t>{-1, static_cast<int64_t>(batch), static_cast<int64_t>(hidden_size)}
                : std::vector<int64_t>{-1, static_cast<int64_t>(hidden_size)};
        output = std::make_shared<op::v1::Reshape>(
            moe,
            op::v0::Constant::create(element::i64, Shape{output_shape.size()}, output_shape),
            true);
    }
    return std::make_shared<ov::Model>(ov::OutputVector{output}, ov::ParameterVector{input});
}

// ============================================================================
// Post-BGM model builders (3 BGMs + compact routing) — Multiply instead of Reshape
// before Unsqueeze (gemma4 pattern: layernorm Multiply feeds expert path directly)
// ============================================================================

inline std::shared_ptr<ov::Model> build_3gemm_bgm_model_multiply_input(
    ov::op::internal::MOE::Activation_type activation_type = ov::op::internal::MOE::Activation_type::SWIGLU) {
    using namespace ov;

    const size_t batch = 2;
    const Dimension in_dim = Dimension::dynamic();
    const size_t hidden_size = 2048;
    const size_t intermediate_size = 4096;
    const size_t number_of_experts = 3;
    const size_t topk = 2;

    auto input = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{batch, in_dim, hidden_size});

    // Simulate layernorm output: Multiply(input, scale) — no Reshape
    auto norm_scale = op::v0::Constant::create(element::f32, Shape{1, 1, hidden_size}, {1.0f});
    auto layernorm_mul = std::make_shared<op::v1::Multiply>(input, norm_scale);

    // Reshape to [batch*seq, hidden] for router path
    auto experts_reshape = std::make_shared<op::v1::Reshape>(
        layernorm_mul,
        op::v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{-1, static_cast<int64_t>(hidden_size)}),
        false);

    // Unsqueeze feeds from Multiply (via Reshape) — but the pattern's optional<Reshape>
    // means hidden_states_m captures the Multiply output (layernorm_mul).
    auto unsqueeze =
        std::make_shared<op::v0::Unsqueeze>(experts_reshape, op::v0::Constant::create(element::i32, Shape{}, {0}));

    // Router subgraph
    auto router_matmul = std::make_shared<op::v0::MatMul>(
        experts_reshape,
        op::v0::Constant::create(element::f32, Shape{number_of_experts, hidden_size}, {1.0f}),
        false,
        true);
    auto router_topk = std::make_shared<op::v11::TopK>(router_matmul,
                                                       op::v0::Constant::create(element::i64, Shape{}, {topk}),
                                                       -1,
                                                       op::v11::TopK::Mode::MAX,
                                                       op::v11::TopK::SortType::SORT_VALUES,
                                                       element::i64);
    auto topk_indices = router_topk->output(1);
    auto chosen_experts = router_topk->output(0);

    // Weights
    auto gate_w =
        op::v0::Constant::create(element::f32, Shape{number_of_experts, intermediate_size, hidden_size}, {1.0f});
    auto up_w =
        op::v0::Constant::create(element::f32, Shape{number_of_experts, intermediate_size, hidden_size}, {1.0f});
    auto down_w =
        op::v0::Constant::create(element::f32, Shape{number_of_experts, hidden_size, intermediate_size}, {1.0f});

    // 3 BGMs
    auto bgm_gate = std::make_shared<GatherMatmul>(unsqueeze, gate_w, topk_indices);
    std::shared_ptr<ov::Node> gate_act;
    if (activation_type == ov::op::internal::MOE::Activation_type::GEGLU_TANH) {
        gate_act = std::make_shared<op::v7::Gelu>(bgm_gate, ov::op::GeluApproximationMode::TANH);
    } else if (activation_type == ov::op::internal::MOE::Activation_type::GEGLU_ERF) {
        gate_act = std::make_shared<op::v7::Gelu>(bgm_gate, ov::op::GeluApproximationMode::ERF);
    } else {
        gate_act = std::make_shared<op::v4::Swish>(bgm_gate);
    }
    auto bgm_up = std::make_shared<GatherMatmul>(unsqueeze, up_w, topk_indices);
    auto swiglu = std::make_shared<op::v1::Multiply>(gate_act, bgm_up);
    auto bgm_down = std::make_shared<GatherMatmul>(swiglu, down_w, topk_indices);

    // Compact routing
    auto router_transpose = std::make_shared<op::v1::Transpose>(
        chosen_experts,
        op::v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{1, 0}));
    auto router_unsqueeze =
        std::make_shared<op::v0::Unsqueeze>(router_transpose, op::v0::Constant::create(element::i32, Shape{}, {-1}));

    // Final: Multiply → ReduceSum → Reshape
    auto final_mul = std::make_shared<op::v1::Multiply>(bgm_down, router_unsqueeze);
    auto reduce_sum =
        std::make_shared<op::v1::ReduceSum>(final_mul,
                                            op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{0}),
                                            false);

    auto end_reshape = std::make_shared<op::v1::Reshape>(
        reduce_sum,
        op::v0::Constant::create(
            element::i64,
            Shape{3},
            std::vector<int64_t>{static_cast<int64_t>(batch), -1, static_cast<int64_t>(hidden_size)}),
        true);

    return std::make_shared<ov::Model>(ov::OutputVector{end_reshape}, ov::ParameterVector{input});
}

inline std::shared_ptr<ov::Model> build_3gemm_bgm_to_moe_reference_model_multiply_input(
    ov::op::internal::MOE::Activation_type activation_type = ov::op::internal::MOE::Activation_type::SWIGLU) {
    using namespace ov;

    const size_t batch = 2;
    const Dimension in_dim = Dimension::dynamic();
    const size_t hidden_size = 2048;
    const size_t intermediate_size = 4096;
    const size_t number_of_experts = 3;
    const size_t topk = 2;

    auto input = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{batch, in_dim, hidden_size});

    // Layernorm Multiply — its OUTPUT is the correct hidden_states for the MOE op
    auto norm_scale = op::v0::Constant::create(element::f32, Shape{1, 1, hidden_size}, {1.0f});
    auto layernorm_mul = std::make_shared<op::v1::Multiply>(input, norm_scale);

    // Router subgraph (not fused, remains in the graph)
    auto experts_reshape = std::make_shared<op::v1::Reshape>(
        layernorm_mul,
        op::v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{-1, static_cast<int64_t>(hidden_size)}),
        false);

    auto router_matmul = std::make_shared<op::v0::MatMul>(
        experts_reshape,
        op::v0::Constant::create(element::f32, Shape{number_of_experts, hidden_size}, {1.0f}),
        false,
        true);
    auto router_topk = std::make_shared<op::v11::TopK>(router_matmul,
                                                       op::v0::Constant::create(element::i64, Shape{}, {topk}),
                                                       -1,
                                                       op::v11::TopK::Mode::MAX,
                                                       op::v11::TopK::SortType::SORT_VALUES,
                                                       element::i64);
    auto topk_indices = router_topk->output(1);
    auto routing = router_topk->output(0);

    // Weights
    auto gate_w =
        op::v0::Constant::create(element::f32, Shape{number_of_experts, intermediate_size, hidden_size}, {1.0f});
    auto up_w =
        op::v0::Constant::create(element::f32, Shape{number_of_experts, intermediate_size, hidden_size}, {1.0f});
    auto down_w =
        op::v0::Constant::create(element::f32, Shape{number_of_experts, hidden_size, intermediate_size}, {1.0f});

    // MOE op — hidden_states input is the layernorm Multiply output (NOT the Parameter)
    ov::OutputVector moe_inputs = {layernorm_mul, routing, topk_indices, gate_w, up_w, down_w};
    ov::op::internal::MOE::Config config;
    config.expert_type = ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU;
    config.activation_type = activation_type;
    auto moe = std::make_shared<ov::op::internal::MOE>(moe_inputs, config);

    return std::make_shared<ov::Model>(ov::OutputVector{moe}, ov::ParameterVector{input});
}

// ============================================================================
// Tests for BGM→MOE passes (Convert3GatherMatmulMoeBlockToMoeOp, Convert2GatherMatmulMoeBlockToMoeOp)
// ============================================================================

TEST_F(TransformationTestsF, Convert3GatherMatmulMoeBlockToMoeOp_basic) {
    model = build_3gemm_bgm_model();
    manager.register_pass<ov::pass::Convert3GatherMatmulMoeBlockToMoeOp>();
    model_ref = build_3gemm_bgm_to_moe_reference_model();
}

TEST_F(TransformationTestsF, Convert3GatherMatmulMoeBlockToMoeOp_without_batch_dim) {
    model = build_3gemm_bgm_model(ov::op::internal::MOE::Activation_type::SWIGLU, false);
    manager.register_pass<ov::pass::Convert3GatherMatmulMoeBlockToMoeOp>();
    model_ref = build_3gemm_bgm_to_moe_reference_model(ov::op::internal::MOE::Activation_type::SWIGLU, false);
}

TEST_F(TransformationTestsF, Convert3GatherMatmulMoeBlockToMoeOp_restores_output_shape) {
    model = build_3gemm_bgm_model(ov::op::internal::MOE::Activation_type::SWIGLU, true, true);
    manager.register_pass<ov::pass::Convert3GatherMatmulMoeBlockToMoeOp>();
    model_ref = build_3gemm_bgm_to_moe_reference_model(ov::op::internal::MOE::Activation_type::SWIGLU, true, true);
}

TEST_F(TransformationTestsF, Convert3GatherMatmulMoeBlockToMoeOp_gelu_tanh) {
    using AT = ov::op::internal::MOE::Activation_type;
    model = build_3gemm_bgm_model(AT::GEGLU_TANH);
    manager.register_pass<ov::pass::Convert3GatherMatmulMoeBlockToMoeOp>();
    model_ref = build_3gemm_bgm_to_moe_reference_model(AT::GEGLU_TANH);
}

TEST_F(TransformationTestsF, Convert3GatherMatmulMoeBlockToMoeOp_gelu_erf) {
    using AT = ov::op::internal::MOE::Activation_type;
    model = build_3gemm_bgm_model(AT::GEGLU_ERF);
    manager.register_pass<ov::pass::Convert3GatherMatmulMoeBlockToMoeOp>();
    model_ref = build_3gemm_bgm_to_moe_reference_model(AT::GEGLU_ERF);
}

TEST_F(TransformationTestsF, Convert3GatherMatmulMoeBlockToMoeOp_multiply_input) {
    model = build_3gemm_bgm_model_multiply_input();
    manager.register_pass<ov::pass::Convert3GatherMatmulMoeBlockToMoeOp>();
    model_ref = build_3gemm_bgm_to_moe_reference_model_multiply_input();
}

TEST_F(TransformationTestsF, Convert3GatherMatmulMoeBlockToMoeOp_multiply_input_gelu_tanh) {
    using AT = ov::op::internal::MOE::Activation_type;
    model = build_3gemm_bgm_model_multiply_input(AT::GEGLU_TANH);
    manager.register_pass<ov::pass::Convert3GatherMatmulMoeBlockToMoeOp>();
    model_ref = build_3gemm_bgm_to_moe_reference_model_multiply_input(AT::GEGLU_TANH);
}

TEST_F(TransformationTestsF, Convert2GatherMatmulMoeBlockToMoeOp_basic) {
    model = build_2gemm_bgm_model();
    manager.register_pass<ov::pass::Convert2GatherMatmulMoeBlockToMoeOp>();
    model_ref = build_2gemm_bgm_to_moe_reference_model();
}

TEST_F(TransformationTestsF, Convert2GatherMatmulMoeBlockToMoeOp_without_batch_dim) {
    model = build_2gemm_bgm_model(false);
    manager.register_pass<ov::pass::Convert2GatherMatmulMoeBlockToMoeOp>();
    model_ref = build_2gemm_bgm_to_moe_reference_model(false);
}

TEST_F(TransformationTestsF, Convert2GatherMatmulMoeBlockToMoeOp_restores_output_shape) {
    model = build_2gemm_bgm_model(true, true);
    manager.register_pass<ov::pass::Convert2GatherMatmulMoeBlockToMoeOp>();
    model_ref = build_2gemm_bgm_to_moe_reference_model(true, true);
}

// Per-group u3 zp is converted to u8
TEST_F(TransformationTestsF, Convert3GatherMatmulMoeBlockToMoeOp_compressed_u3_zp_to_u8) {
    using AT = ov::op::internal::MOE::Activation_type;
    using ov::test::utils::DecompressionType;
    model =
        build_3gemm_bgm_model(AT::SWIGLU, true, false, {{ov::element::u3, DecompressionType::full, ov::element::u3}});
    manager.register_pass<ov::pass::Convert3GatherMatmulMoeBlockToMoeOp>();
    model_ref = build_3gemm_bgm_to_moe_reference_model(AT::SWIGLU,
                                                       true,
                                                       false,
                                                       {{ov::element::u3, DecompressionType::full, ov::element::u8}});
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

// Symmetric u3: the absent zp stays absent
TEST_F(TransformationTestsF, Convert3GatherMatmulMoeBlockToMoeOp_compressed_u3_no_zp) {
    using AT = ov::op::internal::MOE::Activation_type;
    const MoeCompression compression{ov::element::u3, ov::test::utils::DecompressionType::empty};
    model = build_3gemm_bgm_model(AT::SWIGLU, true, false, compression);
    manager.register_pass<ov::pass::Convert3GatherMatmulMoeBlockToMoeOp>();
    model_ref = build_3gemm_bgm_to_moe_reference_model(AT::SWIGLU, true, false, compression);
}

// Sub-byte zp of other types keeps its precision
TEST_F(TransformationTestsF, Convert3GatherMatmulMoeBlockToMoeOp_compressed_u4_zp_unchanged) {
    using AT = ov::op::internal::MOE::Activation_type;
    const MoeCompression compression{ov::element::u4, ov::test::utils::DecompressionType::full, ov::element::u4};
    model = build_3gemm_bgm_model(AT::SWIGLU, true, false, compression);
    manager.register_pass<ov::pass::Convert3GatherMatmulMoeBlockToMoeOp>();
    model_ref = build_3gemm_bgm_to_moe_reference_model(AT::SWIGLU, true, false, compression);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

// Tuple parameters: shared weight/ZP precision, ZP shape, and expected fusion result.
using MoeZeroPointTransformCase = std::tuple<ov::element::Type, ov::test::utils::DecompressionType, bool>;

class Convert3GatherMatmulMoeBlockToMoeOpZeroPointTest : public TransformationTestsF,
                                                         public testing::WithParamInterface<MoeZeroPointTransformCase> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MoeZeroPointTransformCase>& info) {
        const auto& [precision, zp_shape, expect_fusion] = info.param;
        return precision.get_type_name() +
               (zp_shape == ov::test::utils::DecompressionType::scalar ? "_scalar_zp" : "_per_group_zp") +
               (expect_fusion ? "_fused" : "_not_fused");
    }

protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        const auto& [precision, zp_shape, expect_fusion] = GetParam();
        const MoeCompression input_compression{precision, zp_shape, precision};

        model = build_3gemm_bgm_model(ov::op::internal::MOE::Activation_type::SWIGLU, true, false, input_compression);
        manager.register_pass<ov::pass::Convert3GatherMatmulMoeBlockToMoeOp>();
        if (expect_fusion) {
            const auto output_zp_type = precision == ov::element::u3 ? ov::element::u8 : precision;
            const MoeCompression output_compression{precision, zp_shape, output_zp_type};
            model_ref = build_3gemm_bgm_to_moe_reference_model(ov::op::internal::MOE::Activation_type::SWIGLU,
                                                               true,
                                                               false,
                                                               output_compression);
        }
    }
};

TEST_P(Convert3GatherMatmulMoeBlockToMoeOpZeroPointTest, ScalarAndPerGroupZp) {}

INSTANTIATE_TEST_SUITE_P(ScalarZpFused,
                         Convert3GatherMatmulMoeBlockToMoeOpZeroPointTest,
                         testing::Combine(testing::Values(ov::element::u3),
                                          testing::Values(ov::test::utils::DecompressionType::scalar),
                                          testing::Values(true)),
                         Convert3GatherMatmulMoeBlockToMoeOpZeroPointTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    ScalarZpNotFused,
    Convert3GatherMatmulMoeBlockToMoeOpZeroPointTest,
    testing::Combine(testing::Values(ov::element::u8, ov::element::u4, ov::element::i8, ov::element::i4),
                     testing::Values(ov::test::utils::DecompressionType::scalar),
                     testing::Values(false)),
    Convert3GatherMatmulMoeBlockToMoeOpZeroPointTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(PerGroupZp,
                         Convert3GatherMatmulMoeBlockToMoeOpZeroPointTest,
                         testing::Combine(testing::Values(ov::element::u8, ov::element::i8),
                                          testing::Values(ov::test::utils::DecompressionType::full),
                                          testing::Values(true)),
                         Convert3GatherMatmulMoeBlockToMoeOpZeroPointTest::getTestCaseName);

TEST_F(TransformationTestsF, Convert3GatherMatmulMoeBlockToMoeOp_compressed_u3_weights_i8_scalar_zp) {
    using AT = ov::op::internal::MOE::Activation_type;
    using ov::test::utils::DecompressionType;
    model =
        build_3gemm_bgm_model(AT::SWIGLU, true, false, {{ov::element::u3, DecompressionType::scalar, ov::element::i8}});
    manager.register_pass<ov::pass::Convert3GatherMatmulMoeBlockToMoeOp>();
    model_ref = build_3gemm_bgm_to_moe_reference_model(AT::SWIGLU,
                                                       true,
                                                       false,
                                                       {{ov::element::u3, DecompressionType::scalar, ov::element::i8}});
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}
