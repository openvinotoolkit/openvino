// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <vector>

#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/manager.hpp"
#include "partitioning/patterns/opt.hpp"

namespace {

using ResultNodes = std::vector<std::shared_ptr<ov::op::v0::Constant>>;

// Build the subgraph matched by PreserveConstDictMatMulAsymm:
//   Const(W:u8) -> Convert(f16) ------+
//   Const(Z:u8) -> Convert(f16) -> Subtract -> Multiply -> [Convert(f32) ->] MatMul -> Result
//   Const(S:f16) ----------------------------+
//   Parameter(act:f32) -------------------------------------------->
// The Convert between Multiply and MatMul is optional (convert_before_matmul controls it).
struct SubgraphNodes {
    std::shared_ptr<ov::op::v0::Constant> qweight;
    std::shared_ptr<ov::op::v0::Constant> qzerop;
    std::shared_ptr<ov::op::v0::Constant> qcoeff;
    std::shared_ptr<ov::Model> model;
};

SubgraphNodes build_asymm_matmul_subgraph(const ov::Shape& weight_shape,
                                          const ov::element::Type& weight_type,
                                          const ov::Shape& zerop_shape,
                                          const ov::Shape& scale_shape,
                                          bool transpose_b,
                                          bool convert_before_matmul = true) {
    auto qweight = std::make_shared<ov::op::v0::Constant>(weight_type, weight_shape, 0);
    auto qzerop = std::make_shared<ov::op::v0::Constant>(weight_type, zerop_shape, 0);
    // When there is no Convert after Multiply, Multiply must produce f32 directly, which
    // requires scale (qcoeff) to be f32 — matching real onnx-converted model behaviour.
    const auto scale_type = convert_before_matmul ? ov::element::f16 : ov::element::f32;
    auto qcoeff = std::make_shared<ov::op::v0::Constant>(scale_type, scale_shape, 1.0f);

    // When convert_before_matmul=true (optimum-intel style):
    //   u8 → Convert(f16) → Subtract → Multiply(f16) → Convert(f32) → MatMul(f32)
    //   scale (qcoeff) is f16
    // When convert_before_matmul=false (onnx-converted style):
    //   u8 → Convert(f32) → Subtract → Multiply(f32) → MatMul(f32)  [no final Convert]
    //   scale (qcoeff) is f32
    const auto mid_type = convert_before_matmul ? ov::element::f16 : ov::element::f32;
    auto cvtw = std::make_shared<ov::op::v0::Convert>(qweight, mid_type);
    auto cvtz = std::make_shared<ov::op::v0::Convert>(qzerop, mid_type);
    auto sub = std::make_shared<ov::op::v1::Subtract>(cvtw, cvtz);
    auto mul = std::make_shared<ov::op::v1::Multiply>(sub, qcoeff);

    std::shared_ptr<ov::Node> mm_weight_input;
    if (convert_before_matmul) {
        mm_weight_input = std::make_shared<ov::op::v0::Convert>(mul, ov::element::f32);
    } else {
        mm_weight_input = mul;
    }

    const size_t IC = transpose_b ? weight_shape[1] : weight_shape[0];
    auto act = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, IC});

    auto mm = std::make_shared<ov::op::v0::MatMul>(act, mm_weight_input, false, transpose_b);
    auto result = std::make_shared<ov::op::v0::Result>(mm);

    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{act});
    return {qweight, qzerop, qcoeff, model};
}

void run_preserve_pattern(std::shared_ptr<ov::Model>& model, ResultNodes& to_keep) {
    ov::pass::Manager manager;
    using namespace ov::npuw::patterns::opt;
    Context ctx;
    ctx.mm_gate = false;
    manager.register_pass<PreserveConstDictMatMulAsymm>(std::ref(ctx), std::ref(to_keep));
    manager.run_passes(model);
}

// ─────────────────────────────────────────────
// Test 1: Standard layout [OC, IC] + scale [OC, 1] + transpose_b=true → 3 consts kept
// ─────────────────────────────────────────────
TEST(OptPatterns, StandardLayout_MatchesAndPreservesConsts) {
    const ov::Shape weight_shape{32064, 3072};
    const ov::Shape scale_shape{32064, 1};
    auto [qweight, qzerop, qcoeff, model] =
        build_asymm_matmul_subgraph(weight_shape, ov::element::u8, weight_shape, scale_shape, /*transpose_b=*/true);

    ResultNodes to_keep;
    run_preserve_pattern(model, to_keep);

    ASSERT_EQ(to_keep.size(), 3u);
    EXPECT_TRUE(std::find(to_keep.begin(), to_keep.end(), qweight) != to_keep.end());
    EXPECT_TRUE(std::find(to_keep.begin(), to_keep.end(), qzerop) != to_keep.end());
    EXPECT_TRUE(std::find(to_keep.begin(), to_keep.end(), qcoeff) != to_keep.end());
}

// ─────────────────────────────────────────────
// Test 2: Pre-transposed layout [IC, OC] + scale [1, OC] + transpose_b=false → 3 consts kept
// ─────────────────────────────────────────────
TEST(OptPatterns, PreTransposedLayout_MatchesAndPreservesConsts) {
    const ov::Shape weight_shape{3072, 32064};
    const ov::Shape scale_shape{1, 32064};
    auto [qweight, qzerop, qcoeff, model] =
        build_asymm_matmul_subgraph(weight_shape, ov::element::u8, weight_shape, scale_shape, /*transpose_b=*/false);

    ResultNodes to_keep;
    run_preserve_pattern(model, to_keep);

    ASSERT_EQ(to_keep.size(), 3u);
    EXPECT_TRUE(std::find(to_keep.begin(), to_keep.end(), qweight) != to_keep.end());
    EXPECT_TRUE(std::find(to_keep.begin(), to_keep.end(), qzerop) != to_keep.end());
    EXPECT_TRUE(std::find(to_keep.begin(), to_keep.end(), qcoeff) != to_keep.end());
}

// ─────────────────────────────────────────────
// Test 3: Wrong scale layout [IC, OC] (per-element, not per-channel) → no match
// ─────────────────────────────────────────────
TEST(OptPatterns, WrongLayout_ScaleNotPerChannel_NoMatch) {
    const ov::Shape weight_shape{3072, 32064};
    const ov::Shape scale_shape{3072, 32064};  // full per-element: neither [OC,1] nor [1,OC]
    auto [qweight, qzerop, qcoeff, model] =
        build_asymm_matmul_subgraph(weight_shape, ov::element::u8, weight_shape, scale_shape, /*transpose_b=*/false);

    ResultNodes to_keep;
    run_preserve_pattern(model, to_keep);

    EXPECT_TRUE(to_keep.empty());
}

// ─────────────────────────────────────────────
// Test 4: Wrong element type (i8 instead of u8) → no match
// ─────────────────────────────────────────────
TEST(OptPatterns, WrongElemType_NotU8_NoMatch) {
    const ov::Shape weight_shape{32064, 3072};
    const ov::Shape scale_shape{32064, 1};
    auto [qweight, qzerop, qcoeff, model] =
        build_asymm_matmul_subgraph(weight_shape, ov::element::i8, weight_shape, scale_shape, /*transpose_b=*/true);

    ResultNodes to_keep;
    run_preserve_pattern(model, to_keep);

    EXPECT_TRUE(to_keep.empty());
}

// ─────────────────────────────────────────────
// Test 5: Pre-transposed layout without Convert before MatMul (onnx-exported model pattern)
// ─────────────────────────────────────────────
TEST(OptPatterns, PreTransposedLayout_NoConvert_MatchesAndPreservesConsts) {
    const ov::Shape weight_shape{3072, 32064};
    const ov::Shape scale_shape{1, 32064};
    auto [qweight, qzerop, qcoeff, model] = build_asymm_matmul_subgraph(weight_shape,
                                                                        ov::element::u8,
                                                                        weight_shape,
                                                                        scale_shape,
                                                                        /*transpose_b=*/false,
                                                                        /*convert_before_matmul=*/false);

    ResultNodes to_keep;
    run_preserve_pattern(model, to_keep);

    ASSERT_EQ(to_keep.size(), 3u);
    EXPECT_TRUE(std::find(to_keep.begin(), to_keep.end(), qweight) != to_keep.end());
    EXPECT_TRUE(std::find(to_keep.begin(), to_keep.end(), qzerop) != to_keep.end());
    EXPECT_TRUE(std::find(to_keep.begin(), to_keep.end(), qcoeff) != to_keep.end());
}

// ─────────────────────────────────────────────
// Test 6: Standard layout without Convert before MatMul
// ─────────────────────────────────────────────
TEST(OptPatterns, StandardLayout_NoConvert_MatchesAndPreservesConsts) {
    const ov::Shape weight_shape{32064, 3072};
    const ov::Shape scale_shape{32064, 1};
    auto [qweight, qzerop, qcoeff, model] = build_asymm_matmul_subgraph(weight_shape,
                                                                        ov::element::u8,
                                                                        weight_shape,
                                                                        scale_shape,
                                                                        /*transpose_b=*/true,
                                                                        /*convert_before_matmul=*/false);

    ResultNodes to_keep;
    run_preserve_pattern(model, to_keep);

    ASSERT_EQ(to_keep.size(), 3u);
    EXPECT_TRUE(std::find(to_keep.begin(), to_keep.end(), qweight) != to_keep.end());
    EXPECT_TRUE(std::find(to_keep.begin(), to_keep.end(), qzerop) != to_keep.end());
    EXPECT_TRUE(std::find(to_keep.begin(), to_keep.end(), qcoeff) != to_keep.end());
}

}  // namespace
