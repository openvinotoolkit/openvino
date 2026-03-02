// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
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
#include "openvino/op/parameter.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
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
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace {
using ov::test::InputShape;

// Routing type – mirrors MOECompressed::RoutingType but without the GPU dependency
enum class RoutingType { SOFTMAX, SIGMOID_BIAS };

//
// Builds a pre-fusion OV model that exercises the full MOE3Gemm transformation pipeline:
//   1. Expert computation: Reshape→Tile→Reshape→3×MatMul(decompressed weights, transpose_b)
//                          →Swish→Multiply→Reshape
//   2. Routing subgraph: MatMul→Softmax/Sigmoid→TopK→normalize→scatter→reshape→unsqueeze
//   3. Combination: Multiply(expert, routing)→ReduceSum
//
// The GPU plugin transformation pipeline converts this into a fused kernel:
//   FuseVectorizedMOE3GEMM       (folds the expert+combination subgraph into op::internal::MOE)
//   ConvertMOEToMOECompressed     (absorbs compressed weights into MOECompressed)
//   FuseMOE3GemmCompressed        (fuses routing subgraph for Softmax routing)
//   FuseMOE3GemmCompressedSigmoid (fuses routing subgraph for Sigmoid+bias routing)
// which collapse the subgraph into a single MOE3GemmFusedCompressed kernel.
//
// Model parameters (kept small for fast execution):
//   tokens (seq) =   4  (SEQ_LEN)
//   hidden_size  = 128  (HIDDEN_SIZE)
//   inter_size   = 128  (INTER_SIZE)
//   num_experts  =   4  (NUM_EXPERTS)
//   top_k        =   2  (TOP_K)
//   group_size   = 128  (GROUP_SIZE)
//
// op::internal::MOE is GPU-only; validate() checks output shape and finiteness only.
//

class MoE3GemmFusionTest : public testing::WithParamInterface<RoutingType>, virtual public ov::test::SubgraphBaseTest {
public:
    static std::string get_test_case_name(const testing::TestParamInfo<RoutingType>& info) {
        return info.param == RoutingType::SIGMOID_BIAS ? "SigmoidBias" : "Softmax";
    }

protected:
    static constexpr size_t SEQ_LEN = 4;
    static constexpr size_t HIDDEN_SIZE = 128;
    static constexpr size_t INTER_SIZE = 128;
    static constexpr size_t NUM_EXPERTS = 4;
    static constexpr size_t TOP_K = 2;
    static constexpr size_t GROUP_SIZE = 128;

    std::shared_ptr<ov::Model> build_model(RoutingType routing_type) {
        namespace op = ov::op;
        using C = op::v0::Constant;

        // ── Input ─────────────────────────────────────────────────────────────
        // Parameter is f16 so that core_configuration sets inference_precision=f16,
        // which is required by the GPU MOE3GemmFusedCompressed kernel.
        // We immediately convert to f32 for the expert-computation MatMuls
        // (whose decompressed weights are also f32).
        auto hidden_states = std::make_shared<op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, HIDDEN_SIZE});
        auto hidden_states_f32 = std::make_shared<op::v0::Convert>(hidden_states, ov::element::f32);

        // ── Weight decompression: u4 → f16 → Subtract(zp) → Multiply(scale) → Reshape → f32 ──
        //
        //   Gate and Up weights:  [ne, inter_size, group_num, group_size]
        //   Down weights:         [ne, hidden_size, group_num2, group_size]
        //   zp/scale suffix shape:  [ne, inter/hidden, group_num, 1]
        //   Reshaped to:          [ne, inter_size, hidden_size] or [ne, hidden_size, inter_size]
        constexpr size_t GROUP_NUM = HIDDEN_SIZE / GROUP_SIZE;  // = 1
        constexpr size_t GROUP_NUM2 = INTER_SIZE / GROUP_SIZE;  // = 1

        // Helper lambda: build decompression chain for one weight tensor
        auto decompress =
            [&](ov::element::Type wtype, ov::Shape w_shape, ov::Shape zp_shape, ov::Shape sc_shape, std::vector<int32_t> reshape_dims) -> ov::Output<ov::Node> {
            auto wei = C::create(wtype, w_shape, {1});
            auto zp = C::create(wtype, zp_shape, {0});
            auto scale = C::create(ov::element::f16, sc_shape, {0.01f});
            auto rs_c = C::create(ov::element::i32, ov::Shape{3}, reshape_dims);

            auto w_f16 = std::make_shared<op::v0::Convert>(wei, ov::element::f16);
            auto zp_f16 = std::make_shared<op::v0::Convert>(zp, ov::element::f16);
            auto sub = std::make_shared<op::v1::Subtract>(w_f16, zp_f16);
            auto mul = std::make_shared<op::v1::Multiply>(sub, scale);
            auto rs = std::make_shared<op::v1::Reshape>(mul, rs_c, false);
            return std::make_shared<op::v0::Convert>(rs, ov::element::f32);
        };

        // Gate (w0): [ne, inter, GROUP_NUM, GROUP_SIZE], reshaped → [ne, inter, hidden]
        auto convert_gate = decompress(ov::element::u4,
                                       ov::Shape{NUM_EXPERTS, INTER_SIZE, GROUP_NUM, GROUP_SIZE},
                                       ov::Shape{NUM_EXPERTS, INTER_SIZE, GROUP_NUM, 1},
                                       ov::Shape{NUM_EXPERTS, INTER_SIZE, GROUP_NUM, 1},
                                       {static_cast<int32_t>(NUM_EXPERTS), static_cast<int32_t>(INTER_SIZE), static_cast<int32_t>(HIDDEN_SIZE)});

        // Up (w1): same layout as gate
        auto convert_up = decompress(ov::element::u4,
                                     ov::Shape{NUM_EXPERTS, INTER_SIZE, GROUP_NUM, GROUP_SIZE},
                                     ov::Shape{NUM_EXPERTS, INTER_SIZE, GROUP_NUM, 1},
                                     ov::Shape{NUM_EXPERTS, INTER_SIZE, GROUP_NUM, 1},
                                     {static_cast<int32_t>(NUM_EXPERTS), static_cast<int32_t>(INTER_SIZE), static_cast<int32_t>(HIDDEN_SIZE)});

        // Down (w2): [ne, hidden, GROUP_NUM2, GROUP_SIZE], reshaped → [ne, hidden, inter]
        auto convert_down = decompress(ov::element::u4,
                                       ov::Shape{NUM_EXPERTS, HIDDEN_SIZE, GROUP_NUM2, GROUP_SIZE},
                                       ov::Shape{NUM_EXPERTS, HIDDEN_SIZE, GROUP_NUM2, 1},
                                       ov::Shape{NUM_EXPERTS, HIDDEN_SIZE, GROUP_NUM2, 1},
                                       {static_cast<int32_t>(NUM_EXPERTS), static_cast<int32_t>(HIDDEN_SIZE), static_cast<int32_t>(INTER_SIZE)});

        // ── Expert computation subgraph (matched by FuseVectorizedMOE3GEMM) ──
        //   Reshape → Tile → Reshape → 3×MatMul(transpose_b=true) → Swish → Multiply → Reshape
        auto experts_reshape =
            std::make_shared<op::v1::Reshape>(hidden_states_f32,
                                              C::create(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{-1, static_cast<int64_t>(HIDDEN_SIZE)}),
                                              false);

        auto tile = std::make_shared<op::v0::Tile>(experts_reshape,
                                                   C::create(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{static_cast<int64_t>(NUM_EXPERTS), 1}));

        auto after_tile_reshape = std::make_shared<op::v1::Reshape>(
            tile,
            C::create(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{static_cast<int64_t>(NUM_EXPERTS), -1, static_cast<int64_t>(HIDDEN_SIZE)}),
            false);

        // Gate MatMul: [ne, seq, hidden] × [ne, inter, hidden]^T → [ne, seq, inter]
        auto gate_matmul = std::make_shared<op::v0::MatMul>(after_tile_reshape, convert_gate, false, true);
        auto swish = std::make_shared<op::v4::Swish>(gate_matmul);

        // Up MatMul: [ne, seq, hidden] × [ne, inter, hidden]^T → [ne, seq, inter]
        auto up_matmul = std::make_shared<op::v0::MatMul>(after_tile_reshape, convert_up, false, true);

        // SwiGLU: Swish(gate) * up
        auto swiglu = std::make_shared<op::v1::Multiply>(swish, up_matmul);

        // Down MatMul: [ne, seq, inter] × [ne, hidden, inter]^T → [ne, seq, hidden]
        auto down_matmul = std::make_shared<op::v0::MatMul>(swiglu, convert_down, false, true);

        // End reshape to 4D for broadcasting with routing weights
        // Softmax routing produces [ne, seq, 1, 1]; Sigmoid produces [ne, 1, seq, 1]
        std::vector<int64_t> end_reshape_dims;
        if (routing_type == RoutingType::SOFTMAX) {
            end_reshape_dims = {static_cast<int64_t>(NUM_EXPERTS), -1, 1, static_cast<int64_t>(HIDDEN_SIZE)};
        } else {
            end_reshape_dims = {static_cast<int64_t>(NUM_EXPERTS), 1, -1, static_cast<int64_t>(HIDDEN_SIZE)};
        }
        auto end_reshape = std::make_shared<op::v1::Reshape>(down_matmul, C::create(ov::element::i64, ov::Shape{4}, end_reshape_dims), false);

        // ── Router MatMul: [seq, hidden] x [hidden, experts] → [seq, experts] ─
        //    Uses hidden_states directly (not experts_reshape) so that
        //    FuseMOE3GemmCompressed can bind hidden_state_m to both
        //    MOECompressed input 0 and the router matmul input 0.
        auto routers = C::create(ov::element::f32, ov::Shape{HIDDEN_SIZE, NUM_EXPERTS}, {0.2f});
        auto routing_weights = std::make_shared<op::v0::MatMul>(hidden_states_f32, routers);

        // ── Routing subgraph ──────────────────────────────────────────────────
        ov::Output<ov::Node> unsqueeze_moe;

        if (routing_type == RoutingType::SOFTMAX) {
            // MatMul → Softmax → TopK → normalize → scatter → reshape → unsqueeze
            auto softmax = std::make_shared<op::v8::Softmax>(routing_weights, 1);

            auto k = C::create(ov::element::i32, ov::Shape{}, {static_cast<int32_t>(TOP_K)});
            auto topk = std::make_shared<op::v11::TopK>(softmax, k, 1, op::v11::TopK::Mode::MAX, op::v11::TopK::SortType::SORT_VALUES);

            // Normalize top-k weights: values / row_sum
            auto reduce_axis = C::create(ov::element::i64, ov::Shape{1}, {1LL});
            auto reduce_sum = std::make_shared<op::v1::ReduceSum>(topk->output(0), reduce_axis, true);
            auto norm = std::make_shared<op::v1::Divide>(topk->output(0), reduce_sum);

            // Dynamic seq_len: ShapeOf(topk_indices)[0] = seq
            auto shape_of = std::make_shared<op::v3::ShapeOf>(topk->output(1));
            auto gi = C::create(ov::element::i64, ov::Shape{}, {0LL});
            auto ga = C::create(ov::element::i64, ov::Shape{}, {0LL});
            auto gather = std::make_shared<op::v8::Gather>(shape_of, gi, ga);  // scalar: seq
            auto usq_axis0 = C::create(ov::element::i64, ov::Shape{1}, {0LL});
            auto usq_seq = std::make_shared<op::v0::Unsqueeze>(gather, usq_axis0);  // [1]{seq}

            // Derive num_experts dynamically from hidden_states shape to prevent
            // constant folding.  In real models this value comes from a ShapeOf chain
            // on a tensor that has both a dynamic seq dimension and num_experts.
            // Here we compute: num_experts = hidden_size / (hidden_size / num_experts).
            auto shape_of_hs = std::make_shared<op::v3::ShapeOf>(hidden_states_f32, ov::element::i64);
            auto gather_hs_dim1 = std::make_shared<op::v8::Gather>(shape_of_hs,
                                                                   C::create(ov::element::i64, ov::Shape{}, {1LL}),
                                                                   C::create(ov::element::i64, ov::Shape{}, {0LL}));  // scalar: HIDDEN_SIZE
            auto ne_dynamic =
                std::make_shared<op::v1::Divide>(gather_hs_dim1, C::create(ov::element::i64, ov::Shape{}, {static_cast<int64_t>(HIDDEN_SIZE / NUM_EXPERTS)}));
            auto ne_axis = C::create(ov::element::i64, ov::Shape{1}, {0LL});
            auto usq_ne = std::make_shared<op::v0::Unsqueeze>(ne_dynamic, ne_axis);  // [1]{ne}

            // broadcast shape: [seq, num_experts]
            auto bc_shape = std::make_shared<op::v0::Concat>(ov::OutputVector{usq_seq, usq_ne}, 0);
            // reshape shape:  [num_experts, seq, 1]
            auto one_c = C::create(ov::element::i64, ov::Shape{1}, {1LL});
            auto rs_shape = std::make_shared<op::v0::Concat>(ov::OutputVector{usq_ne, usq_seq, one_c}, 0);

            auto zero = C::create(ov::element::f32, ov::Shape{1}, {0.f});
            auto bc = std::make_shared<op::v3::Broadcast>(zero, bc_shape);
            auto sc_axis = C::create(ov::element::i64, ov::Shape{1}, {1LL});
            auto scatter = std::make_shared<op::v12::ScatterElementsUpdate>(bc, topk->output(1), norm, sc_axis, op::v12::ScatterElementsUpdate::Reduction::SUM);

            auto tp_order = C::create(ov::element::i64, ov::Shape{2}, {1LL, 0LL});
            auto transpose = std::make_shared<op::v1::Transpose>(scatter, tp_order);  // [ne, seq]
            auto reshape = std::make_shared<op::v1::Reshape>(transpose, rs_shape, false);
            auto usq_axis3 = C::create(ov::element::i64, ov::Shape{1}, {3LL});
            unsqueeze_moe = std::make_shared<op::v0::Unsqueeze>(reshape, usq_axis3);  // [ne, seq, 1, 1]

        } else {
            // MatMul → Sigmoid → Add(bias) → TopK → Convert(i32) → GatherElements
            //       → normalize → Slice → scatter → transpose → reshape → unsqueeze
            auto sigmoid = std::make_shared<op::v0::Sigmoid>(routing_weights);
            auto routing_bias = C::create(ov::element::f32, ov::Shape{1, NUM_EXPERTS}, {0.1f});
            auto sigmoid_add = std::make_shared<op::v1::Add>(sigmoid, routing_bias);

            auto k = C::create(ov::element::i32, ov::Shape{}, {static_cast<int32_t>(TOP_K)});
            auto topk = std::make_shared<op::v11::TopK>(sigmoid_add, k, 1, op::v11::TopK::Mode::MAX, op::v11::TopK::SortType::SORT_VALUES);

            // Convert i64 topk indices → i32 (required by the pattern matcher)
            auto convert_topk = std::make_shared<op::v0::Convert>(topk->output(1), ov::element::i32);
            auto gather_elements = std::make_shared<op::v6::GatherElements>(sigmoid, convert_topk, 1);

            // Normalize: scores / (sum + eps)
            auto reduce_axis = C::create(ov::element::i64, ov::Shape{1}, {1LL});
            auto reduce_sum = std::make_shared<op::v1::ReduceSum>(gather_elements, reduce_axis, true);
            auto eps = C::create(ov::element::f32, ov::Shape{1, 1}, {1e-6f});
            auto add_eps = std::make_shared<op::v1::Add>(reduce_sum, eps);
            auto norm = std::make_shared<op::v1::Divide>(gather_elements, add_eps);

            // Slice to [seq, top_k] for scatter — stop from ShapeOf (dynamic, matches real model)
            auto sl_start = C::create(ov::element::i32, ov::Shape{2}, std::vector<int32_t>{0, 0});
            auto sl_stop = std::make_shared<op::v3::ShapeOf>(convert_topk, ov::element::i32);
            auto sl_step = C::create(ov::element::i32, ov::Shape{2}, std::vector<int32_t>{1, 1});
            auto sl_axes = C::create(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{0LL, 1LL});
            auto slice = std::make_shared<op::v8::Slice>(norm, sl_start, sl_stop, sl_step, sl_axes);

            // scatter into [seq, num_experts] — shape computed dynamically from ShapeOf
            // to prevent constant folding.  In the real model, seq comes from a ShapeOf
            // chain on a dynamic tensor (e.g., attention output or hidden state).
            auto shape_of_rw = std::make_shared<op::v3::ShapeOf>(routing_weights, ov::element::i64);
            auto gather_seq = std::make_shared<op::v8::Gather>(shape_of_rw,
                                                               C::create(ov::element::i64, ov::Shape{}, {0LL}),
                                                               C::create(ov::element::i64, ov::Shape{}, {0LL}));                   // scalar: seq
            auto usq_seq_sig = std::make_shared<op::v0::Unsqueeze>(gather_seq, C::create(ov::element::i64, ov::Shape{1}, {0LL}));  // [1]{seq}
            auto ne_c = C::create(ov::element::i64, ov::Shape{1}, {static_cast<int64_t>(NUM_EXPERTS)});
            auto bc_shape = std::make_shared<op::v0::Concat>(ov::OutputVector{usq_seq_sig, ne_c}, 0);
            auto zero = C::create(ov::element::f32, ov::Shape{}, {0.f});
            auto bc = std::make_shared<op::v3::Broadcast>(zero, bc_shape);
            auto sc_axis = C::create(ov::element::i64, ov::Shape{}, {1LL});
            auto scatter = std::make_shared<op::v12::ScatterElementsUpdate>(bc, convert_topk, slice, sc_axis, op::v12::ScatterElementsUpdate::Reduction::NONE);

            // Reshape to [num_experts, 1, seq, 1]
            auto tp_order = C::create(ov::element::i32, ov::Shape{2}, std::vector<int32_t>{1, 0});
            auto transpose = std::make_shared<op::v1::Transpose>(scatter, tp_order);
            auto rs_ne = C::create(ov::element::i64, ov::Shape{1}, {static_cast<int64_t>(NUM_EXPERTS)});
            auto rs_one = C::create(ov::element::i64, ov::Shape{1}, {1LL});
            auto rs_shape = std::make_shared<op::v0::Concat>(ov::OutputVector{rs_ne, rs_one, usq_seq_sig}, 0);
            auto reshape = std::make_shared<op::v1::Reshape>(transpose, rs_shape, false);
            auto usq_axis = C::create(ov::element::i64, ov::Shape{}, {3LL});
            unsqueeze_moe = std::make_shared<op::v0::Unsqueeze>(reshape, usq_axis);  // [ne, 1, seq, 1]
        }

        // ── Combination: Multiply(expert, routing) → ReduceSum ────────────────
        // This completes the FuseVectorizedMOE3GEMM pattern:
        //   end_reshape × unsqueeze_routing → reduce_sum
        auto mul3 = std::make_shared<op::v1::Multiply>(end_reshape, unsqueeze_moe);
        auto reduce_sum_final = std::make_shared<op::v1::ReduceSum>(mul3, C::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0LL}), false);

        return std::make_shared<ov::Model>(ov::OutputVector{reduce_sum_final}, ov::ParameterVector{hidden_states});
    }

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;
        inType = outType = ov::element::f16;

        InputShape hidden_shape{ov::PartialShape{-1, HIDDEN_SIZE}, {{SEQ_LEN, HIDDEN_SIZE}}};
        init_input_shapes({hidden_shape});

        function = build_model(GetParam());
    }

    void generate_inputs(const std::vector<ov::Shape>& target_input_static_shapes) override {
        inputs.clear();
        ov::test::utils::InputGenerateData in_data;
        in_data.start_from = -1;
        in_data.range = 2;
        in_data.resolution = 32;
        const auto& model_inputs = function->inputs();
        inputs.insert(
            {model_inputs[0].get_node_shared_ptr(), ov::test::utils::create_and_fill_tensor(ov::element::f16, target_input_static_shapes[0], in_data)});
    }
};

TEST_P(MoE3GemmFusionTest, Inference) {
    run();
}

INSTANTIATE_TEST_SUITE_P(smoke_MoE3GemmFusion,
                         MoE3GemmFusionTest,
                         ::testing::Values(RoutingType::SOFTMAX, RoutingType::SIGMOID_BIAS),
                         MoE3GemmFusionTest::get_test_case_name);
}  // namespace
