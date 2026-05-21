// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/activations_scaling.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "common_test_utils/graph_comparator.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/group_normalization.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/mvn.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/manager.hpp"
#include "ov_ops/moe_compressed.hpp"
#include "transformations/common_optimizations/lin_op_sequence_fusion.hpp"
#include "transformations/common_optimizations/nop_elimination.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov;
using namespace testing;

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v6 = ov::op::v6;
namespace v12 = ov::op::v12;
TEST_F(TransformationTestsF, ScaleDownSingleLayerTest) {
    float scale_factor = 128.f;
    {
        auto input = std::make_shared<v0::Parameter>(ov::element::f16, ov::PartialShape{1, 3, 16, 16});
        auto weights_const = v0::Constant::create(ov::element::f16, ov::Shape{3, 3, 3, 3}, {1});
        auto conv = std::make_shared<v1::Convolution>(input,
                                                      weights_const,
                                                      Strides{},
                                                      CoordinateDiff{},
                                                      CoordinateDiff{},
                                                      Strides{});
        auto bias_const = v0::Constant::create(ov::element::f16, ov::Shape{1, 3, 1, 1}, {2.3f});
        auto add = std::make_shared<v1::Add>(conv, bias_const);
        auto convert = std::make_shared<v0::Convert>(add, ov::element::f32);
        auto result = std::make_shared<v0::Result>(convert);

        model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input});
        manager.register_pass<ov::pass::activations_scaling::ScaleDownSingleLayer>(scale_factor, ov::element::f16);
    }
    {
        auto input = std::make_shared<v0::Parameter>(ov::element::f16, ov::PartialShape{1, 3, 16, 16});
        auto weights_const = v0::Constant::create(ov::element::f16, ov::Shape{3, 3, 3, 3}, {1});
        auto scale_down_const = v0::Constant::create(ov::element::f16, ov::Shape{}, {1.f / scale_factor});
        auto scale_down = std::make_shared<v1::Multiply>(input, scale_down_const);
        auto conv = std::make_shared<v1::Convolution>(scale_down,
                                                      weights_const,
                                                      Strides{},
                                                      CoordinateDiff{},
                                                      CoordinateDiff{},
                                                      Strides{});
        auto bias_const = v0::Constant::create(ov::element::f16, ov::Shape{1, 3, 1, 1}, {2.3f});
        auto scale_down_bias = std::make_shared<v1::Multiply>(bias_const, scale_down_const);
        auto add = std::make_shared<v1::Add>(conv, scale_down_bias);
        auto scale_up_const = v0::Constant::create(ov::element::f16, ov::Shape{}, {scale_factor});
        auto scale_up = std::make_shared<v1::Multiply>(add, scale_up_const);
        auto convert = std::make_shared<v0::Convert>(scale_up, ov::element::f32);
        auto result = std::make_shared<v0::Result>(convert);

        model_ref = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, ScaleDownSingleLayerTest_f32) {
    float scale_factor = 128.f;
    {
        auto input = std::make_shared<v0::Parameter>(ov::element::f16, ov::PartialShape{1, 16});
        auto weights_const0 = v0::Constant::create(ov::element::f16, ov::Shape{16, 8}, {1});
        auto matmul0 = std::make_shared<v0::MatMul>(input, weights_const0);
        auto weights_const1 = v0::Constant::create(ov::element::f16, ov::Shape{8, 16}, {1});
        auto matmul1 = std::make_shared<v0::MatMul>(matmul0, weights_const1);
        auto convert = std::make_shared<v0::Convert>(matmul1, ov::element::f32);
        disable_fp16_compression(convert);
        disable_constant_folding(convert);
        auto convert_f16 = std::make_shared<v0::Convert>(convert, ov::element::f16);
        auto result = std::make_shared<v0::Result>(convert_f16);

        model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input});
        manager.register_pass<ov::pass::activations_scaling::ScaleDownSingleLayer>(scale_factor, ov::element::f16);
        manager.register_pass<ov::pass::MultiplyMultiplyFusion>();
        manager.register_pass<ov::pass::EliminateEltwise>();
    }
    {
        auto input = std::make_shared<v0::Parameter>(ov::element::f16, ov::PartialShape{1, 16});
        auto weights_const0 = v0::Constant::create(ov::element::f16, ov::Shape{16, 8}, {1});
        auto scale_down_const = v0::Constant::create(ov::element::f16, ov::Shape{}, {1.f / scale_factor});
        auto scale_down = std::make_shared<v1::Multiply>(input, scale_down_const);
        auto matmul0 = std::make_shared<v0::MatMul>(scale_down, weights_const0);
        auto weights_const1 = v0::Constant::create(ov::element::f16, ov::Shape{8, 16}, {1});
        auto matmul1 = std::make_shared<v0::MatMul>(matmul0, weights_const1);
        auto convert = std::make_shared<v0::Convert>(matmul1, ov::element::f32);
        auto scale_up_const = v0::Constant::create(ov::element::f32, ov::Shape{}, {scale_factor});
        auto scale_up = std::make_shared<v1::Multiply>(convert, scale_up_const);
        auto convert_f16 = std::make_shared<v0::Convert>(scale_up, ov::element::f16);
        auto result = std::make_shared<v0::Result>(convert_f16);

        model_ref = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input});
    }
}

namespace {
// Shapes used for all GEMM2_BIAS_SWIGLU_CLAMP tests.
// Matches the layout expected by CreateMOECompressedOp in src/plugin/ops/moe.cpp.
constexpr size_t kTokens = 4;
constexpr size_t kHiddenSize = 16;
constexpr size_t kInterSize = 32;  // ofm for up projection (2 * swiglu half)
constexpr size_t kNumExperts = 2;
constexpr size_t kTopK = 2;
constexpr size_t kGroupSize = 8;
constexpr size_t kNumGroups = kHiddenSize / kGroupSize;

// Build a GEMM2_BIAS_SWIGLU_CLAMP MOECompressed node with the given has_zp layout.
// Input order (see scale_down_moe_compressed.cpp):
//   0: hidden_states, 1: routing_weights, 2: topk_idx,
//   3: w_up, 4: scale_up, [5: zp_up,] <bias_up>,
//   <w_down>, <scale_down>, [<zp_down>,] <bias_down>
std::shared_ptr<ov::op::internal::MOECompressed> make_moe_compressed_gemm2(const ov::Output<ov::Node>& hidden_states,
                                                                           const ov::Output<ov::Node>& routing_weights,
                                                                           const ov::Output<ov::Node>& topk_idx,
                                                                           const ov::Output<ov::Node>& bias_up,
                                                                           const ov::Output<ov::Node>& bias_down,
                                                                           bool has_zp,
                                                                           float scale_factor = -1.0f) {
    auto w_up = ov::op::v0::Constant::create(element::u4, Shape{kNumExperts, kInterSize, kNumGroups, kGroupSize}, {1});
    auto scale_up = ov::op::v0::Constant::create(element::f16, Shape{kNumExperts, kInterSize, kNumGroups, 1}, {0.01f});
    auto w_down =
        ov::op::v0::Constant::create(element::u4, Shape{kNumExperts, kHiddenSize, kNumGroups, kGroupSize}, {1});
    auto scale_down =
        ov::op::v0::Constant::create(element::f16, Shape{kNumExperts, kHiddenSize, kNumGroups, 1}, {0.01f});

    ov::OutputVector args;
    args.push_back(hidden_states);
    args.push_back(routing_weights);
    args.push_back(topk_idx);
    args.push_back(w_up);
    args.push_back(scale_up);
    if (has_zp) {
        auto zp_up = ov::op::v0::Constant::create(element::u4, Shape{kNumExperts, kInterSize, kNumGroups, 1}, {0});
        args.push_back(zp_up);
    }
    args.push_back(bias_up);
    args.push_back(w_down);
    args.push_back(scale_down);
    if (has_zp) {
        auto zp_down = ov::op::v0::Constant::create(element::u4, Shape{kNumExperts, kHiddenSize, kNumGroups, 1}, {0});
        args.push_back(zp_down);
    }
    args.push_back(bias_down);

    ov::op::internal::MOECompressed::Config config;
    config.expert_type = ov::op::internal::MOE::Expert_type::GEMM2_BIAS_SWIGLU_CLAMP;
    config.hidden_size = kHiddenSize;
    config.inter_size = kInterSize;
    config.num_expert = kNumExperts;
    config.top_k = kTopK;
    config.group_size = kGroupSize;
    config.has_zp = has_zp;
    config.out_type = ov::element::dynamic;
    config.scale_factor = scale_factor;
    return std::make_shared<ov::op::internal::MOECompressed>(args, config);
}

}  // namespace

TEST_F(TransformationTestsF, ScaleDownSingleLayerTest_MOE_NoZp) {
    const float scale_factor = 8.f;
    {
        auto hidden_states = std::make_shared<ov::op::v0::Parameter>(element::f16, PartialShape{kTokens, kHiddenSize});
        auto routing_weights = std::make_shared<ov::op::v0::Parameter>(element::f16, PartialShape{kTokens, kTopK});
        auto topk_idx = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{kTokens, kTopK});
        auto bias_up = ov::op::v0::Constant::create(element::f16, Shape{kNumExperts, 1, kInterSize}, {0.5f});
        auto bias_down = ov::op::v0::Constant::create(element::f16, Shape{kNumExperts, 1, kHiddenSize}, {0.25f});

        auto moe =
            make_moe_compressed_gemm2(hidden_states, routing_weights, topk_idx, bias_up, bias_down, /*has_zp=*/false);
        auto convert = std::make_shared<ov::op::v0::Convert>(moe, element::f32);
        auto result = std::make_shared<ov::op::v0::Result>(convert);
        model = std::make_shared<ov::Model>(ov::ResultVector{result},
                                            ov::ParameterVector{hidden_states, routing_weights, topk_idx});
        manager.register_pass<ov::pass::activations_scaling::ScaleDownSingleLayer>(scale_factor, ov::element::f16);
    }
    {
        auto hidden_states = std::make_shared<ov::op::v0::Parameter>(element::f16, PartialShape{kTokens, kHiddenSize});
        auto routing_weights = std::make_shared<ov::op::v0::Parameter>(element::f16, PartialShape{kTokens, kTopK});
        auto topk_idx = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{kTokens, kTopK});
        auto bias_up = ov::op::v0::Constant::create(element::f16, Shape{kNumExperts, 1, kInterSize}, {0.5f});
        auto bias_down = ov::op::v0::Constant::create(element::f16, Shape{kNumExperts, 1, kHiddenSize}, {0.25f});

        auto scale_down_const = ov::op::v0::Constant::create(element::f16, Shape{}, {1.f / scale_factor});
        auto hidden_scaled = std::make_shared<ov::op::v1::Multiply>(hidden_states, scale_down_const);
        auto bias_up_scaled = std::make_shared<ov::op::v1::Multiply>(bias_up, scale_down_const);
        auto bias_down_scaled = std::make_shared<ov::op::v1::Multiply>(bias_down, scale_down_const);

        auto moe = make_moe_compressed_gemm2(hidden_scaled,
                                             routing_weights,
                                             topk_idx,
                                             bias_up_scaled,
                                             bias_down_scaled,
                                             /*has_zp=*/false,
                                             /*scale_factor=*/scale_factor);

        auto scale_up_const = ov::op::v0::Constant::create(element::f16, Shape{}, {scale_factor});
        auto scale_up = std::make_shared<ov::op::v1::Multiply>(moe, scale_up_const);
        auto convert = std::make_shared<ov::op::v0::Convert>(scale_up, element::f32);
        auto result = std::make_shared<ov::op::v0::Result>(convert);
        model_ref = std::make_shared<ov::Model>(ov::ResultVector{result},
                                                ov::ParameterVector{hidden_states, routing_weights, topk_idx});
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
}

TEST_F(TransformationTestsF, ScaleDownSingleLayerTest_MOE_WithZp) {
    const float scale_factor = 8.f;
    {
        auto hidden_states = std::make_shared<ov::op::v0::Parameter>(element::f16, PartialShape{kTokens, kHiddenSize});
        auto routing_weights = std::make_shared<ov::op::v0::Parameter>(element::f16, PartialShape{kTokens, kTopK});
        auto topk_idx = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{kTokens, kTopK});
        auto bias_up = ov::op::v0::Constant::create(element::f16, Shape{kNumExperts, 1, kInterSize}, {0.5f});
        auto bias_down = ov::op::v0::Constant::create(element::f16, Shape{kNumExperts, 1, kHiddenSize}, {0.25f});

        auto moe =
            make_moe_compressed_gemm2(hidden_states, routing_weights, topk_idx, bias_up, bias_down, /*has_zp=*/true);
        auto convert = std::make_shared<ov::op::v0::Convert>(moe, element::f32);
        auto result = std::make_shared<ov::op::v0::Result>(convert);
        model = std::make_shared<ov::Model>(ov::ResultVector{result},
                                            ov::ParameterVector{hidden_states, routing_weights, topk_idx});
        manager.register_pass<ov::pass::activations_scaling::ScaleDownSingleLayer>(scale_factor, ov::element::f16);
    }
    {
        auto hidden_states = std::make_shared<ov::op::v0::Parameter>(element::f16, PartialShape{kTokens, kHiddenSize});
        auto routing_weights = std::make_shared<ov::op::v0::Parameter>(element::f16, PartialShape{kTokens, kTopK});
        auto topk_idx = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{kTokens, kTopK});
        auto bias_up = ov::op::v0::Constant::create(element::f16, Shape{kNumExperts, 1, kInterSize}, {0.5f});
        auto bias_down = ov::op::v0::Constant::create(element::f16, Shape{kNumExperts, 1, kHiddenSize}, {0.25f});

        auto scale_down_const = ov::op::v0::Constant::create(element::f16, Shape{}, {1.f / scale_factor});
        auto hidden_scaled = std::make_shared<ov::op::v1::Multiply>(hidden_states, scale_down_const);
        auto bias_up_scaled = std::make_shared<ov::op::v1::Multiply>(bias_up, scale_down_const);
        auto bias_down_scaled = std::make_shared<ov::op::v1::Multiply>(bias_down, scale_down_const);

        auto moe = make_moe_compressed_gemm2(hidden_scaled,
                                             routing_weights,
                                             topk_idx,
                                             bias_up_scaled,
                                             bias_down_scaled,
                                             /*has_zp=*/true,
                                             /*scale_factor=*/scale_factor);

        auto scale_up_const = ov::op::v0::Constant::create(element::f16, Shape{}, {scale_factor});
        auto scale_up = std::make_shared<ov::op::v1::Multiply>(moe, scale_up_const);
        auto convert = std::make_shared<ov::op::v0::Convert>(scale_up, element::f32);
        auto result = std::make_shared<ov::op::v0::Result>(convert);
        model_ref = std::make_shared<ov::Model>(ov::ResultVector{result},
                                                ov::ParameterVector{hidden_states, routing_weights, topk_idx});
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
}

TEST_F(TransformationTestsF, EliminateScalarMulTest) {
    double epsilon = 1.f;
    float scale_factor = 8.f;
    {
        auto input = std::make_shared<v0::Parameter>(ov::element::f16, ov::PartialShape{1, 3, 4, 4});
        auto scale_const = v0::Constant::create(ov::element::f16, ov::Shape{1}, {scale_factor});
        auto mul = std::make_shared<v1::Multiply>(input, scale_const);
        auto norm_scale_const = v0::Constant::create(ov::element::f16, ov::Shape{3}, {10});
        auto norm_bias_const = v0::Constant::create(ov::element::f16, ov::Shape{3}, {10});
        auto group_norm = std::make_shared<v12::GroupNormalization>(mul, norm_scale_const, norm_bias_const, 1, epsilon);
        auto convert = std::make_shared<v0::Convert>(group_norm, ov::element::f32);
        auto result = std::make_shared<v0::Result>(convert);

        model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input});
        manager.register_pass<ov::pass::activations_scaling::EliminateScalarMul>();
    }
    {
        auto input = std::make_shared<v0::Parameter>(ov::element::f16, ov::PartialShape{1, 3, 4, 4});
        auto norm_scale_const = v0::Constant::create(ov::element::f16, ov::Shape{3}, {10});
        auto norm_bias_const = v0::Constant::create(ov::element::f16, ov::Shape{3}, {10});
        epsilon /= scale_factor;
        auto group_norm =
            std::make_shared<v12::GroupNormalization>(input, norm_scale_const, norm_bias_const, 1, epsilon);
        auto convert = std::make_shared<v0::Convert>(group_norm, ov::element::f32);
        auto result = std::make_shared<v0::Result>(convert);

        model_ref = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, MoveDownScalarMulTest) {
    {
        auto input0 = std::make_shared<v0::Parameter>(ov::element::f16, ov::PartialShape{6, 12, 10, 24});
        auto scale_const0 = v0::Constant::create(ov::element::f16, ov::Shape{1}, {10});
        auto mul0 = std::make_shared<v1::Multiply>(input0, scale_const0);
        auto input1 = std::make_shared<v0::Parameter>(ov::element::f16, ov::PartialShape{6, 12, 10, 24});
        auto mul1 = std::make_shared<v1::Multiply>(input1, mul0);
        auto convert = std::make_shared<v0::Convert>(mul1, ov::element::f32);
        auto result = std::make_shared<v0::Result>(convert);

        model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input0, input1});
        manager.register_pass<ov::pass::activations_scaling::MoveDownScalarMul>();
    }
    {
        auto input0 = std::make_shared<v0::Parameter>(ov::element::f16, ov::PartialShape{6, 12, 10, 24});
        auto input1 = std::make_shared<v0::Parameter>(ov::element::f16, ov::PartialShape{6, 12, 10, 24});
        auto mul0 = std::make_shared<v1::Multiply>(input0, input1);
        auto scale_const0 = v0::Constant::create(ov::element::f16, ov::Shape{1}, {10});
        auto mul1 = std::make_shared<v1::Multiply>(mul0, scale_const0);
        auto convert = std::make_shared<v0::Convert>(mul1, ov::element::f32);
        auto result = std::make_shared<v0::Result>(convert);

        model_ref = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input0, input1});
    }
}

TEST_F(TransformationTestsF, MulShareTransformationTest) {
    float epsilon = 1.f;
    float scale_factor = 8.f;
    {
        auto input = std::make_shared<v0::Parameter>(ov::element::f16, ov::PartialShape{1, 3, 4, 4});
        auto mvn_axes = v0::Constant::create(ov::element::i32, ov::Shape{3}, {0, 2, 3});
        auto mvn = std::make_shared<v6::MVN>(input, mvn_axes, true, epsilon, ov::op::MVNEpsMode::INSIDE_SQRT);
        auto convert0 = std::make_shared<v0::Convert>(mvn, ov::element::f32);
        auto result0 = std::make_shared<v0::Result>(convert0);
        auto scale_const = v0::Constant::create(ov::element::f16, ov::Shape{1}, {scale_factor});
        auto mul = std::make_shared<v1::Multiply>(input, scale_const);
        auto convert1 = std::make_shared<v0::Convert>(mul, ov::element::f32);
        auto result1 = std::make_shared<v0::Result>(convert1);

        model = std::make_shared<ov::Model>(ov::ResultVector{result0, result1}, ov::ParameterVector{input});
        manager.register_pass<ov::pass::activations_scaling::MulShareTransformation>();
    }
    {
        auto input = std::make_shared<v0::Parameter>(ov::element::f16, ov::PartialShape{1, 3, 4, 4});
        auto scale_const = v0::Constant::create(ov::element::f16, ov::Shape{1}, {scale_factor});
        auto mul = std::make_shared<v1::Multiply>(input, scale_const);
        auto mvn_axes = v0::Constant::create(ov::element::i32, ov::Shape{3}, {0, 2, 3});
        epsilon *= scale_factor * scale_factor;
        auto mvn = std::make_shared<v6::MVN>(mul, mvn_axes, true, epsilon, ov::op::MVNEpsMode::INSIDE_SQRT);
        auto convert0 = std::make_shared<v0::Convert>(mvn, ov::element::f32);
        auto result0 = std::make_shared<v0::Result>(convert0);
        auto convert1 = std::make_shared<v0::Convert>(mul, ov::element::f32);
        auto result1 = std::make_shared<v0::Result>(convert1);

        model_ref = std::make_shared<ov::Model>(ov::ResultVector{result0, result1}, ov::ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}
