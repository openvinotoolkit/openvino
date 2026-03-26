// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// End-to-end tests for GPU MOE transformation pipeline.
//
// Full pipeline (steps 1+2+3+4): IR → MOE3GemmFusedCompressed
//   IR (Tile+MatMuls+ScatterElementsUpdate) → ConvertTiledMoeBlockTo3GatherMatmuls
//   → ConvertGatherMatmulToGatherMatmulCompressed
//   → Convert3GatherMatmulMoeBlockToMoeOp (Or-patterns produce MOECompressed)
//   → FuseMOE3GemmCompressed → MOE3GemmFusedCompressed + Convert(f32)

#include <gtest/gtest.h>

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
#include "openvino/op/moe.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/serialize.hpp"
#include "openvino/pass/visualize_tree.hpp"
#include "ov_ops/moe_compressed.hpp"
#include "plugin/transformations/fuse_moe_3gemm_compressed.hpp"
#include "transformations/common_optimizations/convert_tiled_moe_block_to_gather_matmuls.hpp"
#include "transformations/common_optimizations/moe_op_fusion.hpp"
#include "transformations/op_conversions/convert_gather_matmul_to_compressed.hpp"

using namespace testing;
using namespace ov::intel_gpu;

namespace ov {
namespace test {
namespace intel_gpu {

// ============================================================================
// Shared constants and helpers
// ============================================================================

static constexpr size_t T2_HIDDEN = 128;
static constexpr size_t T2_INTER = 64;
static constexpr size_t T2_EXPERTS = 4;
static constexpr size_t T2_TOPK = 2;
static constexpr size_t T2_GROUP_SIZE = 32;
static constexpr size_t T2_GROUP_COUNT_GATE = T2_HIDDEN / T2_GROUP_SIZE;  // 4
static constexpr size_t T2_GROUP_COUNT_DOWN = T2_INTER / T2_GROUP_SIZE;   // 2

enum class E2ERoutingType { SOFTMAX, SIGMOID_BIAS };

// Build decompression subgraph matching MOE_COMPRESSED_WEIGHT_GEMM3_PATTERN:
//   Constant(u4, 4D) → Convert(f16) → Subtract(Convert(zp, f16)) → Multiply(scale) → Reshape(3D) → Convert(f32)
static std::shared_ptr<Node> make_gemm3_decompressed_weight(Shape weight_shape,    // [experts, N, group_count, group_size]
                                                            Shape scale_zp_shape,  // [experts, N, group_count, 1]
                                                            Shape reshape_3d) {    // [experts, N, K]
    auto wei = op::v0::Constant::create(element::u4, weight_shape, {1});
    auto zp = op::v0::Constant::create(element::u4, scale_zp_shape, {0});
    auto scale = op::v0::Constant::create(element::f16, scale_zp_shape, {0.01f});

    auto w_f16 = std::make_shared<op::v0::Convert>(wei, element::f16);
    auto zp_f16 = std::make_shared<op::v0::Convert>(zp, element::f16);
    auto sub = std::make_shared<op::v1::Subtract>(w_f16, zp_f16);
    auto mul = std::make_shared<op::v1::Multiply>(sub, scale);

    auto reshape = std::make_shared<op::v1::Reshape>(mul, op::v0::Constant::create(element::i64, Shape{reshape_3d.size()}, reshape_3d), false);

    return std::make_shared<op::v0::Convert>(reshape, element::f32);
}

// Build softmax routing IR subgraph:
//   MatMul → Softmax → TopK → ReduceSum → Divide → ScatterElementsUpdate → Transpose → Reshape → Unsqueeze
static void build_softmax_routing_ir(const std::shared_ptr<Node>& experts_reshape,
                                     const std::shared_ptr<Node>& end_reshape,
                                     const std::shared_ptr<Node>& param,
                                     std::shared_ptr<Node>& mul3_out,
                                     std::shared_ptr<Node>& reduce_sum_final_out) {
    auto routers = op::v0::Constant::create(element::f32, Shape{T2_EXPERTS, T2_HIDDEN}, {1.0f});
    auto router_matmul = std::make_shared<op::v0::MatMul>(experts_reshape, routers, false, true);
    auto softmax = std::make_shared<op::v8::Softmax>(router_matmul, 1);
    auto topk = std::make_shared<op::v11::TopK>(softmax,
                                                op::v0::Constant::create(element::i64, Shape{}, {T2_TOPK}),
                                                -1,
                                                op::v11::TopK::Mode::MAX,
                                                op::v11::TopK::SortType::SORT_VALUES,
                                                element::i64);

    auto topk_values = topk->output(0);
    auto topk_indices = topk->output(1);

    // Normalize TopK values
    auto reduce_sum = std::make_shared<op::v1::ReduceSum>(topk_values, op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{-1}), true);
    auto divide = std::make_shared<op::v1::Divide>(topk_values, reduce_sum);

    // ScatterElementsUpdate: broadcast zeros to [tokens, experts], scatter normalized values
    auto zero_const = op::v0::Constant::create(element::f32, Shape{1}, {0});
    auto experts_const = op::v0::Constant::create(element::i64, Shape{1}, {static_cast<int64_t>(T2_EXPERTS)});
    auto topk_shape = std::make_shared<op::v3::ShapeOf>(topk_indices, element::i64);
    auto first_topk_dim = std::make_shared<op::v8::Gather>(topk_shape,
                                                           op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{0}),
                                                           op::v0::Constant::create(element::i64, Shape{}, std::vector<int64_t>{0}));
    auto bcast_shape = std::make_shared<op::v0::Concat>(OutputVector{first_topk_dim, experts_const}, 0);
    auto scatter_data = std::make_shared<op::v3::Broadcast>(zero_const, bcast_shape);
    auto scatter = std::make_shared<op::v12::ScatterElementsUpdate>(scatter_data,
                                                                    topk_indices,
                                                                    divide,
                                                                    op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{1}));

    // Transpose → Reshape → Unsqueeze (routing weights)
    auto router_transpose = std::make_shared<op::v1::Transpose>(scatter, op::v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{1, 0}));
    auto minus_one = op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{-1});
    auto router_shape = std::make_shared<op::v0::Concat>(OutputVector{experts_const, first_topk_dim, minus_one}, 0);
    auto router_reshape = std::make_shared<op::v1::Reshape>(router_transpose, router_shape, true);
    auto routing_unsqueeze =
        std::make_shared<op::v0::Unsqueeze>(router_reshape, op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{-1}));

    // Combine: end_reshape * routing → ReduceSum(axis=0) → output
    mul3_out = std::make_shared<op::v1::Multiply>(end_reshape, routing_unsqueeze);
    reduce_sum_final_out = std::make_shared<op::v1::ReduceSum>(mul3_out, op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{0}), false);
}

// Build sigmoid+bias routing IR subgraph:
//   MatMul → Sigmoid → Add(bias) → TopK → Convert(i32)
//   GatherElements(sigmoid, convert) → ReduceSum → Add(eps) → Divide → Slice
//   → ScatterElementsUpdate → Transpose → Reshape → Unsqueeze
static void build_sigmoid_routing_ir(const std::shared_ptr<Node>& experts_reshape,
                                     const std::shared_ptr<Node>& end_reshape,
                                     const std::shared_ptr<Node>& param,
                                     std::shared_ptr<Node>& mul3_out,
                                     std::shared_ptr<Node>& reduce_sum_final_out) {
    auto routers = op::v0::Constant::create(element::f32, Shape{T2_EXPERTS, T2_HIDDEN}, {1.0f});
    auto router_matmul = std::make_shared<op::v0::MatMul>(experts_reshape, routers, false, true);

    // Sigmoid → Add(bias) → TopK
    auto sigmoid = std::make_shared<op::v0::Sigmoid>(router_matmul);
    auto routing_bias = op::v0::Constant::create(element::f32, Shape{1, T2_EXPERTS}, {0.1f});
    auto sig_add = std::make_shared<op::v1::Add>(sigmoid, routing_bias);

    auto topk = std::make_shared<op::v11::TopK>(sig_add,
                                                op::v0::Constant::create(element::i64, Shape{}, {T2_TOPK}),
                                                -1,
                                                op::v11::TopK::Mode::MAX,
                                                op::v11::TopK::SortType::SORT_VALUES,
                                                element::i64);

    auto topk_indices = topk->output(1);
    auto convert_topk = std::make_shared<op::v0::Convert>(topk_indices, element::i32);

    // GatherElements(sigmoid, convert) → ReduceSum → Add(eps) → Divide → Slice
    auto gather_el = std::make_shared<op::v6::GatherElements>(sigmoid, convert_topk, 1);
    auto reduce_sum = std::make_shared<op::v1::ReduceSum>(gather_el,
                                                          op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{1}),
                                                          true);
    auto eps = op::v0::Constant::create(element::f32, Shape{1, 1}, {1e-6f});
    auto add_eps = std::make_shared<op::v1::Add>(reduce_sum, eps);
    auto norm = std::make_shared<op::v1::Divide>(gather_el, add_eps);

    // Slice (normalize to topk shape)
    auto sl_stop = std::make_shared<op::v3::ShapeOf>(convert_topk, element::i32);
    auto slice = std::make_shared<op::v8::Slice>(norm,
                                                 op::v0::Constant::create(element::i32, Shape{2}, std::vector<int32_t>{0, 0}),
                                                 sl_stop,
                                                 op::v0::Constant::create(element::i32, Shape{2}, std::vector<int32_t>{1, 1}),
                                                 op::v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{0, 1}));

    // ScatterElementsUpdate
    auto zero_const = op::v0::Constant::create(element::f32, Shape{1}, {0});
    auto experts_const = op::v0::Constant::create(element::i64, Shape{1}, {static_cast<int64_t>(T2_EXPERTS)});
    auto topk_shape = std::make_shared<op::v3::ShapeOf>(convert_topk, element::i64);
    auto first_topk_dim = std::make_shared<op::v8::Gather>(topk_shape,
                                                           op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{0}),
                                                           op::v0::Constant::create(element::i64, Shape{}, std::vector<int64_t>{0}));
    auto bcast_shape = std::make_shared<op::v0::Concat>(OutputVector{first_topk_dim, experts_const}, 0);
    auto scatter_data = std::make_shared<op::v3::Broadcast>(zero_const, bcast_shape);
    auto scatter = std::make_shared<op::v12::ScatterElementsUpdate>(scatter_data,
                                                                    ov::Output<ov::Node>(convert_topk),
                                                                    slice,
                                                                    op::v0::Constant::create(element::i64, Shape{}, std::vector<int64_t>{1}));

    // Transpose → Reshape → Unsqueeze
    auto router_transpose = std::make_shared<op::v1::Transpose>(scatter,
                                                                op::v0::Constant::create(element::i32, Shape{2}, std::vector<int32_t>{1, 0}));
    auto minus_one = op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{-1});
    auto one_const = op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{1});
    auto router_shape = std::make_shared<op::v0::Concat>(OutputVector{experts_const, one_const, first_topk_dim}, 0);
    auto router_reshape = std::make_shared<op::v1::Reshape>(router_transpose, router_shape, true);
    auto routing_unsqueeze =
        std::make_shared<op::v0::Unsqueeze>(router_reshape, op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{-1}));

    // Combine: end_reshape * routing → ReduceSum(axis=0) → output
    mul3_out = std::make_shared<op::v1::Multiply>(end_reshape, routing_unsqueeze);
    reduce_sum_final_out = std::make_shared<op::v1::ReduceSum>(mul3_out, op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{0}), false);
}

// ============================================================================
// Full IR → MOE3GemmFusedCompressed pipeline (parameterized by routing type)
// ============================================================================

class MOE_E2E_PipelineTest : public TransformationTestsF,
                             public ::testing::WithParamInterface<E2ERoutingType> {
public:
    static std::string get_test_case_name(const ::testing::TestParamInfo<E2ERoutingType>& info) {
        return info.param == E2ERoutingType::SOFTMAX ? "Softmax" : "SigmoidBias";
    }
};

TEST_P(MOE_E2E_PipelineTest, IRToMOE3GemmFusedCompressed) {
    const auto routing_type = GetParam();
    disable_rt_info_check();
    {
        // ---- Build input model: IR with 3 MatMuls + Tile + decompressed weights ----
        auto param = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1, static_cast<int64_t>(T2_HIDDEN)});

        // Reshape (3D→2D) → Tile → Reshape for expert expansion
        auto experts_reshape =
            std::make_shared<op::v1::Reshape>(param,
                                              op::v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{-1, static_cast<int64_t>(T2_HIDDEN)}),
                                              false);
        auto tile = std::make_shared<op::v0::Tile>(experts_reshape,
                                                   op::v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{static_cast<int64_t>(T2_EXPERTS), 1}));
        auto after_tile_reshape = std::make_shared<op::v1::Reshape>(
            tile,
            op::v0::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{static_cast<int64_t>(T2_EXPERTS), -1, static_cast<int64_t>(T2_HIDDEN)}),
            false);

        // Weight decompression chains (u4 → f16 → Sub(zp) → Mul(scale) → Reshape(3D) → Convert(f32))
        auto gate_w = make_gemm3_decompressed_weight({T2_EXPERTS, T2_INTER, T2_GROUP_COUNT_GATE, T2_GROUP_SIZE},
                                                     {T2_EXPERTS, T2_INTER, T2_GROUP_COUNT_GATE, 1},
                                                     {T2_EXPERTS, T2_INTER, T2_HIDDEN});
        auto up_w = make_gemm3_decompressed_weight({T2_EXPERTS, T2_INTER, T2_GROUP_COUNT_GATE, T2_GROUP_SIZE},
                                                   {T2_EXPERTS, T2_INTER, T2_GROUP_COUNT_GATE, 1},
                                                   {T2_EXPERTS, T2_INTER, T2_HIDDEN});
        auto down_w = make_gemm3_decompressed_weight({T2_EXPERTS, T2_HIDDEN, T2_GROUP_COUNT_DOWN, T2_GROUP_SIZE},
                                                     {T2_EXPERTS, T2_HIDDEN, T2_GROUP_COUNT_DOWN, 1},
                                                     {T2_EXPERTS, T2_HIDDEN, T2_INTER});

        // 3 GEMMs: gate → Swish, up, SwiGLU = Swish * up, down
        auto gate_matmul = std::make_shared<op::v0::MatMul>(after_tile_reshape, gate_w, false, true);
        auto swish = std::make_shared<op::v4::Swish>(gate_matmul);
        auto up_matmul = std::make_shared<op::v0::MatMul>(after_tile_reshape, up_w, false, true);
        auto swiglu = std::make_shared<op::v1::Multiply>(swish, up_matmul);
        auto down_matmul = std::make_shared<op::v0::MatMul>(swiglu, down_w, false, true);

        // End reshape: [experts, tokens, hidden] → [experts, tokens, 1, hidden]
        auto end_reshape = std::make_shared<op::v1::Reshape>(
            down_matmul,
            op::v0::Constant::create(element::i64, Shape{4}, std::vector<int64_t>{static_cast<int64_t>(T2_EXPERTS), -1, 1, static_cast<int64_t>(T2_HIDDEN)}),
            false);

        // Build routing subgraph depending on type
        std::shared_ptr<Node> mul3, reduce_sum_final;
        if (routing_type == E2ERoutingType::SOFTMAX) {
            build_softmax_routing_ir(experts_reshape, end_reshape, param, mul3, reduce_sum_final);
        } else {
            build_sigmoid_routing_ir(experts_reshape, end_reshape, param, mul3, reduce_sum_final);
        }

        model = std::make_shared<Model>(reduce_sum_final, ParameterVector{param});

        // Register pipeline passes
        // Step 1: IR → GatherMatmul
        manager.register_pass<ov::pass::ConvertTiledMoeBlockTo3GatherMatmuls>();
        // Step 2: GatherMatmul → GatherMatmulCompressed
        manager.register_pass<ov::pass::ConvertGatherMatmulToGatherMatmulCompressed>(
            std::vector<ov::element::Type>{ov::element::f32, ov::element::f16, ov::element::i8, ov::element::u8},
            std::vector<ov::element::Type>{ov::element::f16, ov::element::u4, ov::element::i4, ov::element::i8, ov::element::u8});
        // Step 3: GatherMatmul/GatherMatmulCompressed → MOE/MOECompressed
        manager.register_pass<ov::pass::Convert3GatherMatmulMoeBlockToMoeOp>(/*has_batch_dim=*/1);
        // Step 4: MOECompressed → MOE3GemmFusedCompressed
        manager.register_pass<FuseMOE3GemmCompressed>();
    }
    {
        // ---- Reference model: MOE3GemmFusedCompressed → Convert(f32) ----
        auto param = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1, static_cast<int64_t>(T2_HIDDEN)});

        auto experts_reshape =
            std::make_shared<op::v1::Reshape>(param,
                                              op::v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{-1, static_cast<int64_t>(T2_HIDDEN)}),
                                              false);
        auto routers = op::v0::Constant::create(element::f32, Shape{T2_EXPERTS, T2_HIDDEN}, {1.0f});
        auto routing_weights = std::make_shared<op::v0::MatMul>(experts_reshape, routers, false, true);

        // Weights after process_compressed_weights: groups combined to 3D [experts, N, K]
        auto wei_gate = op::v0::Constant::create(element::u4, Shape{T2_EXPERTS, T2_INTER, T2_HIDDEN}, {1});
        auto wei_up = op::v0::Constant::create(element::u4, Shape{T2_EXPERTS, T2_INTER, T2_HIDDEN}, {1});
        auto wei_down = op::v0::Constant::create(element::u4, Shape{T2_EXPERTS, T2_HIDDEN, T2_INTER}, {1});

        // Scales after process_compressed_weights: trailing 1 combined → 3D [experts, N, groups]
        auto scale_gate = op::v0::Constant::create(element::f16, Shape{T2_EXPERTS, T2_INTER, T2_GROUP_COUNT_GATE}, {0.01f});
        auto scale_up = op::v0::Constant::create(element::f16, Shape{T2_EXPERTS, T2_INTER, T2_GROUP_COUNT_GATE}, {0.01f});
        auto scale_down = op::v0::Constant::create(element::f16, Shape{T2_EXPERTS, T2_HIDDEN, T2_GROUP_COUNT_DOWN}, {0.01f});

        // Zero-points after process_compressed_weights: trailing 1 combined → 3D
        auto zp_gate = op::v0::Constant::create(element::u4, Shape{T2_EXPERTS, T2_INTER, T2_GROUP_COUNT_GATE}, {0});
        auto zp_up = op::v0::Constant::create(element::u4, Shape{T2_EXPERTS, T2_INTER, T2_GROUP_COUNT_GATE}, {0});
        auto zp_down = op::v0::Constant::create(element::u4, Shape{T2_EXPERTS, T2_HIDDEN, T2_GROUP_COUNT_DOWN}, {0});

        ov::op::internal::MOECompressed::Config config;
        config.expert_type = ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU;
        config.hidden_size = T2_HIDDEN;
        config.inter_size = T2_INTER;
        config.num_expert = T2_EXPERTS;
        config.group_size = T2_GROUP_SIZE;
        config.top_k = T2_TOPK;
        config.out_type = element::f16;
        config.has_batch_dim = 1;

        // The fused op takes the 2D reshaped input, not the original 3D param
        OutputVector args{experts_reshape, routing_weights, wei_gate, scale_gate, zp_gate, wei_up, scale_up, zp_up, wei_down, scale_down, zp_down};

        if (routing_type == E2ERoutingType::SIGMOID_BIAS) {
            config.routing_type = ov::op::internal::MOECompressed::RoutingType::SIGMOID_BIAS;
            auto routing_bias = op::v0::Constant::create(element::f32, Shape{1, T2_EXPERTS}, {0.1f});
            args.push_back(routing_bias);
            auto routing_eps = op::v0::Constant::create(element::f32, Shape{1, 1}, {1e-6f});
            args.push_back(routing_eps);
        }

        std::shared_ptr<Node> moe_fused = std::make_shared<ov::intel_gpu::op::MOE3GemmFusedCompressed>(args, config);

        // MOECompressed's input was the original 3D param, so the transformation
        // inserts ShapeOf(param) → Reshape to restore the original shape
        auto hidden_state_shape = std::make_shared<op::v3::ShapeOf>(param);
        moe_fused = std::make_shared<op::v1::Reshape>(moe_fused, hidden_state_shape, false);

        // Original MOE output was f32, MOE3GemmFusedCompressed out_type=f16 → needs Convert(f32)
        auto convert_back = std::make_shared<op::v0::Convert>(moe_fused, element::f32);

        model_ref = std::make_shared<Model>(convert_back, ParameterVector{param});
    }
}

INSTANTIATE_TEST_SUITE_P(smoke,
                         MOE_E2E_PipelineTest,
                         ::testing::Values(E2ERoutingType::SOFTMAX, E2ERoutingType::SIGMOID_BIAS),
                         MOE_E2E_PipelineTest::get_test_case_name);

}  // namespace intel_gpu
}  // namespace test
}  // namespace ov
