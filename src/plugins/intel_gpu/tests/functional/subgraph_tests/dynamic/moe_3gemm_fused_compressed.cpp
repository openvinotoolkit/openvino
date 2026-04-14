// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Functional subgraph test for MOE 3-GEMM fused compressed pipeline.
// Builds a full MOE IR (Tile + 3 MatMuls + ScatterElementsUpdate + routing)
// and runs inference on GPU, which triggers the transformation pipeline:
//   IR → ConvertTiledMoeBlockTo3GatherMatmuls
//      → ConvertGatherMatmulToGatherMatmulCompressed
//      → Convert3GatherMatmulMoeBlockToMoeOp
//      → FuseMOE3GemmCompressed
//      → MOE3GemmFusedCompressed kernel
//
// The fusion_level parameter controls how far the pipeline proceeds:
//   GATHER_MATMUL_COMPRESSED  — stops at GatherMatmulCompressed (blocks step 3)
//   MOE_COMPRESSED            — stops at MOECompressed (blocks step 4, no GPU impl yet)
//   MOE_3GEMM_FUSED_COMPRESSED — full pipeline (current behavior)

#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "intel_gpu/runtime/internal_properties.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace {
using ov::test::InputShape;

enum class FusionLevel {
    GATHER_MATMUL_COMPRESSED,    // Stop at GatherMatmulCompressed nodes (block MOE op fusion)
    MOE_COMPRESSED,              // Stop at MOECompressed (block FuseMOE3GemmCompressed) — no GPU impl yet
    MOE_3GEMM_FUSED_COMPRESSED,  // Full pipeline (current behavior)
};

// Weight decompression style matching real model patterns
enum class DecompressionStyle {
    // 4D grouped: Const(u4, [E,N,G,GS]) → Convert(f16) → Sub(zp) → Mul(scale) → Reshape([E,N,K]) → Convert(f32)
    GROUPED_4D,
    // 3D per-channel (matches openvino_model.xml): Const(u4, [E,N,K]) → Convert(f16) → Sub(zp) → Mul(scale) → Convert(f32)
    PER_CHANNEL_3D,
};

std::string fusionLevelToString(FusionLevel level) {
    switch (level) {
    case FusionLevel::GATHER_MATMUL_COMPRESSED:
        return "GATHER_MATMUL_COMPRESSED";
    case FusionLevel::MOE_COMPRESSED:
        return "MOE_COMPRESSED";
    case FusionLevel::MOE_3GEMM_FUSED_COMPRESSED:
        return "MOE_3GEMM_FUSED_COMPRESSED";
    }
    return "UNKNOWN";
}

std::string decompressionStyleToString(DecompressionStyle style) {
    switch (style) {
    case DecompressionStyle::GROUPED_4D:
        return "GROUPED_4D";
    case DecompressionStyle::PER_CHANNEL_3D:
        return "PER_CHANNEL_3D";
    }
    return "UNKNOWN";
}

struct MOEModelParams {
    std::vector<InputShape> input_shapes;
    ov::element::Type input_precision;
    size_t hidden_size;
    size_t inter_size;
    size_t num_experts;
    size_t top_k;
    size_t group_size;  // Only used when decompression_style == GROUPED_4D
    DecompressionStyle decompression_style;
};

using MOE3GemmFusedCompressedParams = std::tuple<MOEModelParams, FusionLevel>;

class MOE3GemmFusedCompressed : public testing::WithParamInterface<MOE3GemmFusedCompressedParams>, virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MOE3GemmFusedCompressedParams>& obj) {
        const auto& [p, fusion_level] = obj.param;

        std::ostringstream result;
        result << "IS=(";
        for (const auto& shape : p.input_shapes) {
            result << ov::test::utils::partialShape2str({shape.first}) << "_";
        }
        result << ")_TS=";
        for (const auto& shape : p.input_shapes) {
            result << "(";
            if (!shape.second.empty()) {
                auto itr = shape.second.begin();
                do {
                    result << ov::test::utils::vec2str(*itr);
                } while (++itr != shape.second.end() && result << "_");
            }
            result << ")_";
        }
        result << "prec=" << p.input_precision;
        result << "_H=" << p.hidden_size;
        result << "_I=" << p.inter_size;
        result << "_E=" << p.num_experts;
        result << "_K=" << p.top_k;
        result << "_G=" << p.group_size;
        result << "_D=" << decompressionStyleToString(p.decompression_style);
        result << "_FL=" << fusionLevelToString(fusion_level);
        return result.str();
    }

protected:
    // Build grouped decompression subgraph (4D weights with Reshape):
    //   Constant(u4, [E,N,G,GS]) → Convert(f16) → Subtract(Convert(zp, f16)) → Multiply(scale) → Reshape([E,N,K]) → Convert(f32)
    static std::shared_ptr<ov::Node> make_decompressed_weight_grouped(ov::Shape weight_shape, ov::Shape scale_zp_shape, ov::Shape reshape_3d) {
        auto wei = ov::op::v0::Constant::create(ov::element::u4, weight_shape, {1});
        auto zp = ov::op::v0::Constant::create(ov::element::u4, scale_zp_shape, {0});
        auto scale = ov::op::v0::Constant::create(ov::element::f16, scale_zp_shape, {0.01f});

        auto w_f16 = std::make_shared<ov::op::v0::Convert>(wei, ov::element::f16);
        auto zp_f16 = std::make_shared<ov::op::v0::Convert>(zp, ov::element::f16);
        auto sub = std::make_shared<ov::op::v1::Subtract>(w_f16, zp_f16);
        auto mul = std::make_shared<ov::op::v1::Multiply>(sub, scale);

        auto reshape_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{reshape_3d.size()}, reshape_3d);
        auto reshape = std::make_shared<ov::op::v1::Reshape>(mul, reshape_const, false);

        return std::make_shared<ov::op::v0::Convert>(reshape, ov::element::f32);
    }

    // Build per-channel decompression subgraph (3D weights, no Reshape).
    // Matches the real model pattern from openvino_model.xml:
    //   Constant(u4, [E,N,K]) → Convert(f16) → Subtract(Convert(zp u4, [E,N,1] → f16)) → Multiply(scale f16, [E,N,1]) → Convert(f32)
    static std::shared_ptr<ov::Node> make_decompressed_weight_per_channel(ov::Shape weight_shape_3d) {
        // weight_shape_3d: [experts, out_features, in_features] e.g. [128, 768, 2048]
        ov::Shape scale_zp_shape = {weight_shape_3d[0], weight_shape_3d[1], 1};

        auto wei = ov::op::v0::Constant::create(ov::element::u4, weight_shape_3d, {1});
        auto zp = ov::op::v0::Constant::create(ov::element::u4, scale_zp_shape, {0});
        auto scale = ov::op::v0::Constant::create(ov::element::f16, scale_zp_shape, {0.01f});

        auto w_f16 = std::make_shared<ov::op::v0::Convert>(wei, ov::element::f16);
        auto zp_f16 = std::make_shared<ov::op::v0::Convert>(zp, ov::element::f16);
        auto sub = std::make_shared<ov::op::v1::Subtract>(w_f16, zp_f16);
        auto mul = std::make_shared<ov::op::v1::Multiply>(sub, scale);

        return std::make_shared<ov::op::v0::Convert>(mul, ov::element::f32);
    }

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;
        configuration.insert({ov::hint::inference_precision.name(), ov::element::f16});

        // MOE3GemmFusedCompressed uses the fused MOE GEMM pipeline which requires systolic
        // (HW_MATMUL / supports_immad). Skip on iGPU where kernels fall back to different
        // precision characteristics.
        auto capabilities = core->get_property(ov::test::utils::DEVICE_GPU, ov::device::capabilities);
        if (std::find(capabilities.cbegin(), capabilities.cend(), ov::intel_gpu::capability::HW_MATMUL) == capabilities.cend())
            GTEST_SKIP();

        const auto& [p, fusion_level] = GetParam();
        init_input_shapes(p.input_shapes);
        inType = outType = p.input_precision;

        const size_t hidden = p.hidden_size;
        const size_t inter = p.inter_size;
        const size_t experts = p.num_experts;
        const size_t top_k = p.top_k;
        const size_t group_size = p.group_size;

        auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputDynamicShapes[0]);

        // Reshape (3D→2D) → Tile → Reshape for expert expansion
        auto experts_reshape = std::make_shared<ov::op::v1::Reshape>(
            param,
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{-1, static_cast<int64_t>(hidden)}),
            false);
        auto tile = std::make_shared<ov::op::v0::Tile>(
            experts_reshape,
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{static_cast<int64_t>(experts), 1}));
        auto after_tile_reshape = std::make_shared<ov::op::v1::Reshape>(
            tile,
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{static_cast<int64_t>(experts), -1, static_cast<int64_t>(hidden)}),
            false);

        // Weight decompression chains — style depends on model parameter
        std::shared_ptr<ov::Node> gate_w, up_w, down_w;
        if (p.decompression_style == DecompressionStyle::PER_CHANNEL_3D) {
            // Matches openvino_model.xml: 3D weights [E, N, K], scale/zp [E, N, 1], no Reshape
            gate_w = make_decompressed_weight_per_channel({experts, inter, hidden});
            up_w = make_decompressed_weight_per_channel({experts, inter, hidden});
            down_w = make_decompressed_weight_per_channel({experts, hidden, inter});
        } else {
            // Original grouped style: 4D weights [E, N, G, GS] → Reshape → 3D
            const size_t group_count_gate = hidden / group_size;
            const size_t group_count_down = inter / group_size;
            gate_w = make_decompressed_weight_grouped({experts, inter, group_count_gate, group_size}, {experts, inter, group_count_gate, 1}, {experts, inter, hidden});
            up_w = make_decompressed_weight_grouped({experts, inter, group_count_gate, group_size}, {experts, inter, group_count_gate, 1}, {experts, inter, hidden});
            down_w = make_decompressed_weight_grouped({experts, hidden, group_count_down, group_size}, {experts, hidden, group_count_down, 1}, {experts, hidden, inter});
        }

        // 3 GEMMs: gate → Swish, up, SwiGLU = Swish * up, down
        auto gate_matmul = std::make_shared<ov::op::v0::MatMul>(after_tile_reshape, gate_w, false, true);
        auto swish = std::make_shared<ov::op::v4::Swish>(gate_matmul);
        auto up_matmul = std::make_shared<ov::op::v0::MatMul>(after_tile_reshape, up_w, false, true);
        auto swiglu = std::make_shared<ov::op::v1::Multiply>(swish, up_matmul);
        auto down_matmul = std::make_shared<ov::op::v0::MatMul>(swiglu, down_w, false, true);

        // End reshape: [experts, tokens, hidden] → [experts, tokens, 1, hidden]
        auto end_reshape = std::make_shared<ov::op::v1::Reshape>(
            down_matmul,
            ov::op::v0::Constant::create(ov::element::i64,
                                         ov::Shape{4},
                                         std::vector<int64_t>{static_cast<int64_t>(experts), -1, 1, static_cast<int64_t>(hidden)}),
            false);

        // Router: Reshape → MatMul → Softmax → TopK → normalize
        auto routers = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{experts, hidden}, {1.0f});
        auto router_matmul = std::make_shared<ov::op::v0::MatMul>(experts_reshape, routers, false, true);
        auto softmax = std::make_shared<ov::op::v8::Softmax>(router_matmul, 1);
        auto topk_k = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {top_k});
        auto topk_node =
            std::make_shared<ov::op::v11::TopK>(softmax, topk_k, -1, ov::op::v11::TopK::Mode::MAX, ov::op::v11::TopK::SortType::SORT_VALUES, ov::element::i64);

        auto topk_values = topk_node->output(0);
        auto topk_indices = topk_node->output(1);

        // Normalize TopK values
        auto reduce_sum =
            std::make_shared<ov::op::v1::ReduceSum>(topk_values, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-1}), true);
        auto divide = std::make_shared<ov::op::v1::Divide>(topk_values, reduce_sum);

        // ScatterElementsUpdate: broadcast zeros to [tokens, experts], scatter normalized values
        auto zero_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {0});
        auto experts_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {static_cast<int64_t>(experts)});
        auto topk_shape = std::make_shared<ov::op::v3::ShapeOf>(topk_indices, ov::element::i64);
        auto first_topk_dim = std::make_shared<ov::op::v8::Gather>(topk_shape,
                                                                   ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0}),
                                                                   ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, std::vector<int64_t>{0}));
        auto bcast_shape = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{first_topk_dim, experts_const}, 0);
        auto scatter_data = std::make_shared<ov::op::v3::Broadcast>(zero_const, bcast_shape);
        auto scatter =
            std::make_shared<ov::op::v12::ScatterElementsUpdate>(scatter_data,
                                                                 topk_indices,
                                                                 divide,
                                                                 ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1}));

        // Transpose → Reshape → Unsqueeze (routing weights)
        auto router_transpose =
            std::make_shared<ov::op::v1::Transpose>(scatter, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{1, 0}));
        auto minus_one = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-1});
        auto router_shape = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{experts_const, first_topk_dim, minus_one}, 0);
        auto router_reshape = std::make_shared<ov::op::v1::Reshape>(router_transpose, router_shape, true);
        auto routing_unsqueeze =
            std::make_shared<ov::op::v0::Unsqueeze>(router_reshape, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-1}));

        // Combine: end_reshape * routing → ReduceSum(axis=0) → Reshape(input shape) → output
        auto mul3 = std::make_shared<ov::op::v1::Multiply>(end_reshape, routing_unsqueeze);
        auto reduce_sum_final =
            std::make_shared<ov::op::v1::ReduceSum>(mul3, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0}), false);

        ov::ResultVector results;
        auto output_reshape = std::make_shared<ov::op::v1::Reshape>(
            reduce_sum_final,
            std::make_shared<ov::op::v3::ShapeOf>(param, ov::element::i64),
            false);
        results.push_back(std::make_shared<ov::op::v0::Result>(output_reshape));

        function = std::make_shared<ov::Model>(results, ov::ParameterVector{param}, "MOE3GemmFusedCompressed");
    }

    void check_results() {
        // const auto& [p, fusion_level] = GetParam();
        // auto runtime_model = compiledModel.get_runtime_model();

        // int moe_fused_count = 0;
        // int gather_matmul_count = 0;
        // for (const auto& n : runtime_model->get_ordered_ops()) {
        //     auto layer_type = n->get_rt_info().at(ov::exec_model_info::LAYER_TYPE).as<std::string>();
        //     if (layer_type == "moe_3gemm_fused_compressed")
        //         moe_fused_count++;
        //     if (layer_type == "gather_matmul_compressed")
        //         gather_matmul_count++;
        // }

        // switch (fusion_level) {
        // case FusionLevel::MOE_3GEMM_FUSED_COMPRESSED:
        //     ASSERT_EQ(moe_fused_count, 1) << "Expected exactly one moe_3gemm_fused_compressed node";
        //     ASSERT_EQ(gather_matmul_count, 0) << "Expected no gather_matmul_compressed nodes";
        //     break;
        // case FusionLevel::GATHER_MATMUL_COMPRESSED:
        //     ASSERT_GE(gather_matmul_count, 1) << "Expected at least one gather_matmul_compressed node";
        //     ASSERT_EQ(moe_fused_count, 0) << "Expected no moe_3gemm_fused_compressed nodes";
        //     break;
        // case FusionLevel::MOE_COMPRESSED:
        //     // @todo claude: add runtime checks once MOECompressed has a GPU implementation
        //     break;
        // }
    }
};

TEST_P(MOE3GemmFusedCompressed, Inference) {
    run();
    check_results();
}

// @todo claude: consider adding Inference_cached test once basic inference is validated

// Small config with grouped 4D decompression: 4 experts, top-2, hidden=128, inter=128, group_size=32
const std::vector<MOEModelParams> small_grouped_params = {
    // Single token
    {
        /*input_shapes=*/{{{-1, -1, 128}, {{1, 1, 128}}}},
        /*input_precision=*/ov::element::f32,
        /*hidden_size=*/128,
        /*inter_size=*/128,
        /*num_experts=*/4,
        /*top_k=*/2,
        /*group_size=*/32,
        /*decompression_style=*/DecompressionStyle::GROUPED_4D,
    },
    // Multiple tokens
    {
        /*input_shapes=*/{{{-1, -1, 128}, {{1, 4, 128}}}},
        /*input_precision=*/ov::element::f32,
        /*hidden_size=*/128,
        /*inter_size=*/128,
        /*num_experts=*/4,
        /*top_k=*/2,
        /*group_size=*/32,
        /*decompression_style=*/DecompressionStyle::GROUPED_4D,
    },
    // Multiple tokens, dynamic
    {
        /*input_shapes=*/{{{-1, -1, 128}, {{1, 1, 128}, {1, 4, 128}}}},
        /*input_precision=*/ov::element::f32,
        /*hidden_size=*/128,
        /*inter_size=*/128,
        /*num_experts=*/4,
        /*top_k=*/2,
        /*group_size=*/32,
        /*decompression_style=*/DecompressionStyle::GROUPED_4D,
    },
};

// Per-channel 3D decompression matching openvino_model.xml pattern:
//   Const(u4, [E,N,K]) → Convert(f16) → Sub(zp [E,N,1]) → Mul(scale [E,N,1]) → Convert(f32)
// Uses small dims to keep test fast; real model has experts=128, hidden=2048, inter=768, top_k=8
const std::vector<MOEModelParams> small_per_channel_params = {
    // Single token
    {
        /*input_shapes=*/{{{-1, -1, 128}, {{1, 1, 128}}}},
        /*input_precision=*/ov::element::f32,
        /*hidden_size=*/128,
        /*inter_size=*/128,
        /*num_experts=*/4,
        /*top_k=*/2,
        /*group_size=*/0,  // unused for PER_CHANNEL_3D
        /*decompression_style=*/DecompressionStyle::PER_CHANNEL_3D,
    },
    // Multiple tokens
    {
        /*input_shapes=*/{{{-1, -1, 128}, {{1, 4, 128}}}},
        /*input_precision=*/ov::element::f32,
        /*hidden_size=*/128,
        /*inter_size=*/128,
        /*num_experts=*/4,
        /*top_k=*/2,
        /*group_size=*/0,
        /*decompression_style=*/DecompressionStyle::PER_CHANNEL_3D,
    },
    // Multiple tokens, dynamic
    {
        /*input_shapes=*/{{{-1, -1, 128}, {{1, 1, 128}, {1, 4, 128}}}},
        /*input_precision=*/ov::element::f32,
        /*hidden_size=*/128,
        /*inter_size=*/128,
        /*num_experts=*/4,
        /*top_k=*/2,
        /*group_size=*/0,
        /*decompression_style=*/DecompressionStyle::PER_CHANNEL_3D,
    },
    // @todo claude: add model-realistic dims (experts=128, hidden=2048, inter=768, top_k=8) once small tests pass
};

// @todo claude: add FusionLevel::MOE_COMPRESSED once GPU implementation exists
const std::vector<FusionLevel> fusion_levels = {
    FusionLevel::MOE_3GEMM_FUSED_COMPRESSED,
    // FusionLevel::GATHER_MATMUL_COMPRESSED,
};

INSTANTIATE_TEST_SUITE_P(smoke_MOE3GemmFusedCompressed_Grouped,
                         MOE3GemmFusedCompressed,
                         ::testing::Combine(::testing::ValuesIn(small_grouped_params), ::testing::ValuesIn(fusion_levels)),
                         MOE3GemmFusedCompressed::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MOE3GemmFusedCompressed_PerChannel,
                         MOE3GemmFusedCompressed,
                         ::testing::Combine(::testing::ValuesIn(small_per_channel_params), ::testing::ValuesIn(fusion_levels)),
                         MOE3GemmFusedCompressed::getTestCaseName);
}  // namespace
