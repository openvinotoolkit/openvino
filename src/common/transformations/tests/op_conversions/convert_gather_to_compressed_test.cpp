// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_gather_to_compressed.hpp"

#include <memory>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/manager.hpp"
#include "ov_ops/gather_compressed.hpp"

using namespace testing;
using namespace ov::pass;

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v8 = ov::op::v8;
namespace v11 = ov::op::v11;
TEST_F(TransformationTestsF, ConvertGatherToCompressed1) {
    {
        auto input1 = std::make_shared<v0::Parameter>(ov::element::i32, ov::PartialShape{-1, 16});
        auto axis_const = v0::Constant::create(ov::element::i32, ov::Shape{1}, {1});
        auto weights_const = v0::Constant::create(ov::element::u8, ov::Shape{32, 16}, {1});
        auto convert = std::make_shared<v0::Convert>(weights_const, ov::element::f32);
        auto scale_const = v0::Constant::create(ov::element::f32, ov::Shape{32, 1}, {1});
        auto scale = std::make_shared<v1::Multiply>(convert, scale_const);
        auto gather = std::make_shared<v8::Gather>(scale, input1, axis_const);

        model = std::make_shared<ov::Model>(ov::OutputVector{gather}, ov::ParameterVector{input1});
        manager.register_pass<ConvertGatherToGatherCompressed>();
    }
    {
        auto input1 = std::make_shared<v0::Parameter>(ov::element::i32, ov::PartialShape{-1, 16});
        auto axis_const = v0::Constant::create(ov::element::i32, ov::Shape{1}, {1});
        auto weights_const = v0::Constant::create(ov::element::u8, ov::Shape{32, 16}, {1});
        auto scale_const = v0::Constant::create(ov::element::f32, ov::Shape{32, 1}, {1});
        auto gather_compressed =
            std::make_shared<ov::op::internal::GatherCompressed>(weights_const, input1, axis_const, 0, scale_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{gather_compressed}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertGatherToCompressed2) {
    {
        auto input1 = std::make_shared<v0::Parameter>(ov::element::i32, ov::PartialShape{-1, 16});
        auto axis_const = v0::Constant::create(ov::element::i32, ov::Shape{1}, {1});
        auto weights_const = v0::Constant::create(ov::element::u8, ov::Shape{32, 16}, {1});
        auto convert = std::make_shared<v0::Convert>(weights_const, ov::element::f32);
        auto zp_const = v0::Constant::create(ov::element::f32, ov::Shape{32, 1}, {1});
        auto sub = std::make_shared<v1::Subtract>(convert, zp_const);
        auto scale_const = v0::Constant::create(ov::element::f32, ov::Shape{32, 1}, {1});
        auto scale = std::make_shared<v1::Multiply>(sub, scale_const);
        auto gather = std::make_shared<v8::Gather>(scale, input1, axis_const);

        model = std::make_shared<ov::Model>(ov::OutputVector{gather}, ov::ParameterVector{input1});
        manager.register_pass<ConvertGatherToGatherCompressed>();
    }
    {
        auto input1 = std::make_shared<v0::Parameter>(ov::element::i32, ov::PartialShape{-1, 16});
        auto axis_const = v0::Constant::create(ov::element::i32, ov::Shape{1}, {1});
        auto weights_const = v0::Constant::create(ov::element::u8, ov::Shape{32, 16}, {1});
        auto scale_const = v0::Constant::create(ov::element::f32, ov::Shape{32, 1}, {1});
        auto zp_const = v0::Constant::create(ov::element::f32, ov::Shape{32, 1}, {1});
        auto gather_compressed = std::make_shared<ov::op::internal::GatherCompressed>(weights_const,
                                                                                      input1,
                                                                                      axis_const,
                                                                                      0,
                                                                                      scale_const,
                                                                                      zp_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{gather_compressed}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertGatherToCompressed3) {
    {
        auto input1 = std::make_shared<v0::Parameter>(ov::element::i32, ov::PartialShape{-1, 16});
        auto axis_const = v0::Constant::create(ov::element::i32, ov::Shape{1}, {1});
        auto weights_const = v0::Constant::create(ov::element::u8, ov::Shape{32, 4, 4}, {1});
        auto convert = std::make_shared<v0::Convert>(weights_const, ov::element::f32);
        auto zp_const = v0::Constant::create(ov::element::f32, ov::Shape{32, 4, 1}, {1});
        auto sub = std::make_shared<v1::Subtract>(convert, zp_const);
        auto scale_const = v0::Constant::create(ov::element::f32, ov::Shape{32, 4, 1}, {1});
        auto scale = std::make_shared<v1::Multiply>(sub, scale_const);
        auto reshape_const = v0::Constant::create(ov::element::i32, ov::Shape{2}, {-1, 16});
        auto reshape = std::make_shared<v1::Reshape>(scale, reshape_const, false);
        auto gather = std::make_shared<v8::Gather>(reshape, input1, axis_const);

        model = std::make_shared<ov::Model>(ov::OutputVector{gather}, ov::ParameterVector{input1});
        manager.register_pass<ConvertGatherToGatherCompressed>();
    }
    {
        auto input1 = std::make_shared<v0::Parameter>(ov::element::i32, ov::PartialShape{-1, 16});
        auto axis_const = v0::Constant::create(ov::element::i32, ov::Shape{1}, {1});
        auto weights_const = v0::Constant::create(ov::element::u8, ov::Shape{32, 16}, {1});
        auto scale_const = v0::Constant::create(ov::element::f32, ov::Shape{32, 4}, {1});
        auto zp_const = v0::Constant::create(ov::element::f32, ov::Shape{32, 4}, {1});
        auto gather_compressed = std::make_shared<ov::op::internal::GatherCompressed>(weights_const,
                                                                                      input1,
                                                                                      axis_const,
                                                                                      0,
                                                                                      scale_const,
                                                                                      zp_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{gather_compressed}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertGatherToCompressed4) {
    {
        auto input1 = std::make_shared<v0::Parameter>(ov::element::i32, ov::PartialShape{-1, 16});
        auto axis_const = v0::Constant::create(ov::element::i32, ov::Shape{1}, {1});
        auto weights_const = v0::Constant::create(ov::element::u4, ov::Shape{32, 4, 4}, {1});
        auto convert = std::make_shared<v0::Convert>(weights_const, ov::element::f32);
        auto zp_const = v0::Constant::create(ov::element::f32, ov::Shape{1, 1, 1}, {1});
        auto sub = std::make_shared<v1::Subtract>(convert, zp_const);
        auto scale_const = v0::Constant::create(ov::element::f32, ov::Shape{32, 4, 1}, {1});
        auto scale = std::make_shared<v1::Multiply>(sub, scale_const);
        auto reshape_const = v0::Constant::create(ov::element::i32, ov::Shape{2}, {-1, 16});
        auto reshape = std::make_shared<v1::Reshape>(scale, reshape_const, false);
        auto gather = std::make_shared<v8::Gather>(reshape, input1, axis_const);

        model = std::make_shared<ov::Model>(ov::OutputVector{gather}, ov::ParameterVector{input1});
        manager.register_pass<ConvertGatherToGatherCompressed>();
    }
    {
        auto input1 = std::make_shared<v0::Parameter>(ov::element::i32, ov::PartialShape{-1, 16});
        auto axis_const = v0::Constant::create(ov::element::i32, ov::Shape{1}, {1});
        auto weights_const = v0::Constant::create(ov::element::u4, ov::Shape{32, 16}, {1});
        auto scale_const = v0::Constant::create(ov::element::f32, ov::Shape{32, 4}, {1});
        auto zp_const = v0::Constant::create(ov::element::f32, ov::Shape{1, 1}, {1});
        auto gather_compressed = std::make_shared<ov::op::internal::GatherCompressed>(weights_const,
                                                                                      input1,
                                                                                      axis_const,
                                                                                      0,
                                                                                      scale_const,
                                                                                      zp_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{gather_compressed}, ov::ParameterVector{input1});
    }
}

namespace {
// GatherCompressed requires each scale / zero point value to cover a contiguous block of the flattened weights,
// so the decompression subgraph must be kept as is for the other layouts.
std::shared_ptr<ov::Model> make_gather_decompression_model(const ov::element::Type& weights_type,
                                                           const ov::Shape& weights_shape,
                                                           const ov::Shape& scale_shape,
                                                           const ov::Shape& zp_shape = {}) {
    auto input1 = std::make_shared<v0::Parameter>(ov::element::i32, ov::PartialShape{-1, 16});
    auto axis_const = v0::Constant::create(ov::element::i32, ov::Shape{1}, {1});
    auto weights_const = v0::Constant::create(weights_type, weights_shape, {1});
    std::shared_ptr<ov::Node> decompression = std::make_shared<v0::Convert>(weights_const, ov::element::f32);
    if (!zp_shape.empty()) {
        auto zp_const = v0::Constant::create(ov::element::f32, zp_shape, {1});
        decompression = std::make_shared<v1::Subtract>(decompression, zp_const);
    }
    auto scale_const = v0::Constant::create(ov::element::f32, scale_shape, {1});
    decompression = std::make_shared<v1::Multiply>(decompression, scale_const);
    if (weights_shape.size() == 3) {
        auto reshape_const = v0::Constant::create(ov::element::i32, ov::Shape{2}, {-1, 16});
        decompression = std::make_shared<v1::Reshape>(decompression, reshape_const, false);
    }
    auto gather = std::make_shared<v8::Gather>(decompression, input1, axis_const);
    return std::make_shared<ov::Model>(ov::OutputVector{gather}, ov::ParameterVector{input1});
}
}  // namespace

using ConvertGatherToCompressedNegativeParams = std::tuple<ov::element::Type,  // weights type
                                                           ov::Shape,          // weights shape
                                                           ov::Shape,          // scale shape
                                                           ov::Shape>;         // zero point shape, empty if none

class ConvertGatherToCompressedNegativeTests
    : public testing::WithParamInterface<ConvertGatherToCompressedNegativeParams>,
      public TransformationTestsF {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConvertGatherToCompressedNegativeParams>& info) {
        const auto& [weights_type, weights_shape, scale_shape, zp_shape] = info.param;
        std::ostringstream ss;
        ss << "weights=" << weights_type << weights_shape << "_scale=" << scale_shape << "_zp=" << zp_shape;
        return ss.str();
    }

protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        const auto& [weights_type, weights_shape, scale_shape, zp_shape] = GetParam();
        model = make_gather_decompression_model(weights_type, weights_shape, scale_shape, zp_shape);
        manager.register_pass<ConvertGatherToGatherCompressed>();
    }
};

TEST_P(ConvertGatherToCompressedNegativeTests, NotConverted) {}

INSTANTIATE_TEST_SUITE_P(ConvertGatherToCompressed_Negative,
                         ConvertGatherToCompressedNegativeTests,
                         ::testing::Values(
                             // per-column scale
                             ConvertGatherToCompressedNegativeParams{ov::element::u8, {32, 16}, {1, 16}, {}},
                             ConvertGatherToCompressedNegativeParams{ov::element::u4, {32, 16}, {1, 16}, {}},
                             // per-column zero point
                             ConvertGatherToCompressedNegativeParams{ov::element::u8, {32, 16}, {32, 1}, {1, 16}},
                             // scale that doesn't vary along the leading weights dim
                             ConvertGatherToCompressedNegativeParams{ov::element::u8, {32, 4, 4}, {1, 4, 1}, {}},
                             // scale that broadcasts the weights
                             ConvertGatherToCompressedNegativeParams{ov::element::u8, {32, 1}, {32, 4}, {}}),
                         ConvertGatherToCompressedNegativeTests::getTestCaseName);

TEST_F(TransformationTestsF, ConvertGatherToCompressedU2NotMatchedByDefault) {
    // u2 is not in the default type list, so the graph must stay as is.
    auto input1 = std::make_shared<v0::Parameter>(ov::element::i32, ov::PartialShape{-1, 16});
    auto axis_const = v0::Constant::create(ov::element::i32, ov::Shape{1}, {1});
    auto weights_const = v0::Constant::create(ov::element::u2, ov::Shape{32, 16}, {1});
    auto convert = std::make_shared<v0::Convert>(weights_const, ov::element::f32);
    auto scale_const = v0::Constant::create(ov::element::f32, ov::Shape{32, 1}, {1});
    auto scale = std::make_shared<v1::Multiply>(convert, scale_const);
    auto gather = std::make_shared<v8::Gather>(scale, input1, axis_const);

    model = std::make_shared<ov::Model>(ov::OutputVector{gather}, ov::ParameterVector{input1});
    manager.register_pass<ConvertGatherToGatherCompressed>();
}

TEST_F(TransformationTestsF, ConvertGatherToCompressedU2) {
    {
        auto input1 = std::make_shared<v0::Parameter>(ov::element::i32, ov::PartialShape{-1, 16});
        auto axis_const = v0::Constant::create(ov::element::i32, ov::Shape{1}, {1});
        auto weights_const = v0::Constant::create(ov::element::u2, ov::Shape{32, 16}, {1});
        auto convert = std::make_shared<v0::Convert>(weights_const, ov::element::f32);
        auto zp_const = v0::Constant::create(ov::element::f32, ov::Shape{32, 1}, {1});
        auto sub = std::make_shared<v1::Subtract>(convert, zp_const);
        auto scale_const = v0::Constant::create(ov::element::f32, ov::Shape{32, 1}, {1});
        auto scale = std::make_shared<v1::Multiply>(sub, scale_const);
        auto gather = std::make_shared<v8::Gather>(scale, input1, axis_const);

        model = std::make_shared<ov::Model>(ov::OutputVector{gather}, ov::ParameterVector{input1});
        manager.register_pass<ConvertGatherToGatherCompressed>(std::vector<ov::element::Type>{ov::element::u8,
                                                                                              ov::element::i8,
                                                                                              ov::element::u4,
                                                                                              ov::element::i4,
                                                                                              ov::element::u2});
    }
    {
        auto input1 = std::make_shared<v0::Parameter>(ov::element::i32, ov::PartialShape{-1, 16});
        auto axis_const = v0::Constant::create(ov::element::i32, ov::Shape{1}, {1});
        auto weights_const = v0::Constant::create(ov::element::u2, ov::Shape{32, 16}, {1});
        auto scale_const = v0::Constant::create(ov::element::f32, ov::Shape{32, 1}, {1});
        auto zp_const = v0::Constant::create(ov::element::f32, ov::Shape{32, 1}, {1});
        auto gather_compressed = std::make_shared<ov::op::internal::GatherCompressed>(weights_const,
                                                                                      input1,
                                                                                      axis_const,
                                                                                      0,
                                                                                      scale_const,
                                                                                      zp_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{gather_compressed}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertGatherToCompressedFP16) {
    {
        auto input1 = std::make_shared<v0::Parameter>(ov::element::i32, ov::PartialShape{-1, 16});
        auto axis_const = v0::Constant::create(ov::element::i32, ov::Shape{1}, {1});
        auto weights_const = v0::Constant::create(ov::element::u8, ov::Shape{32, 16}, {1});
        auto convert = std::make_shared<v0::Convert>(weights_const, ov::element::f16);
        auto zp_const = v0::Constant::create(ov::element::f16, ov::Shape{32, 1}, {1});
        auto sub = std::make_shared<v1::Subtract>(convert, zp_const);
        auto scale_const = v0::Constant::create(ov::element::f16, ov::Shape{32, 1}, {1});
        auto scale = std::make_shared<v1::Multiply>(sub, scale_const);
        auto scale_convert = std::make_shared<v0::Convert>(scale, ov::element::f32);
        auto gather = std::make_shared<v8::Gather>(scale_convert, input1, axis_const);

        model = std::make_shared<ov::Model>(ov::OutputVector{gather}, ov::ParameterVector{input1});
        manager.register_pass<ConvertGatherToGatherCompressed>();
    }
    {
        auto input1 = std::make_shared<v0::Parameter>(ov::element::i32, ov::PartialShape{-1, 16});
        auto axis_const = v0::Constant::create(ov::element::i32, ov::Shape{1}, {1});
        auto weights_const = v0::Constant::create(ov::element::u8, ov::Shape{32, 16}, {1});
        auto scale_const = v0::Constant::create(ov::element::f16, ov::Shape{32, 1}, {1});
        auto scale_convert = std::make_shared<v0::Convert>(scale_const, ov::element::f32);
        auto zp_const = v0::Constant::create(ov::element::f16, ov::Shape{32, 1}, {1});
        auto gather_compressed = std::make_shared<ov::op::internal::GatherCompressed>(weights_const,
                                                                                      input1,
                                                                                      axis_const,
                                                                                      0,
                                                                                      scale_convert,
                                                                                      zp_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{gather_compressed}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertGatherToCompressedMultiOutput) {
    {
        auto input1 = std::make_shared<v0::Parameter>(ov::element::i32, ov::PartialShape{-1, 16});

        auto input2 = std::make_shared<v0::Parameter>(ov::element::i32, ov::Shape{});
        auto topk =
            std::make_shared<v11::TopK>(input1, input2, 0, v11::TopK::Mode::MAX, v11::TopK::SortType::SORT_VALUES);

        auto axis_const = v0::Constant::create(ov::element::i32, ov::Shape{1}, {1});
        auto weights_const = v0::Constant::create(ov::element::u8, ov::Shape{32, 16}, {1});
        auto convert = std::make_shared<v0::Convert>(weights_const, ov::element::f32);
        auto scale_const = v0::Constant::create(ov::element::f32, ov::Shape{32, 1}, {1});
        auto scale = std::make_shared<v1::Multiply>(convert, scale_const);
        auto gather = std::make_shared<v8::Gather>(scale, topk->output(1), axis_const);

        model = std::make_shared<ov::Model>(ov::OutputVector{gather}, ov::ParameterVector{input1, input2});
        manager.register_pass<ConvertGatherToGatherCompressed>();
    }
    {
        auto input1 = std::make_shared<v0::Parameter>(ov::element::i32, ov::PartialShape{-1, 16});

        auto input2 = std::make_shared<v0::Parameter>(ov::element::i32, ov::Shape{});
        auto topk =
            std::make_shared<v11::TopK>(input1, input2, 0, v11::TopK::Mode::MAX, v11::TopK::SortType::SORT_VALUES);

        auto axis_const = v0::Constant::create(ov::element::i32, ov::Shape{1}, {1});
        auto weights_const = v0::Constant::create(ov::element::u8, ov::Shape{32, 16}, {1});
        auto scale_const = v0::Constant::create(ov::element::f32, ov::Shape{32, 1}, {1});
        auto gather_compressed = std::make_shared<ov::op::internal::GatherCompressed>(weights_const,
                                                                                      topk->output(1),
                                                                                      axis_const,
                                                                                      0,
                                                                                      scale_const);

        model_ref =
            std::make_shared<ov::Model>(ov::OutputVector{gather_compressed}, ov::ParameterVector{input1, input2});
    }
}

// In compressed FP16/BF16 weight case, gather node with constant weight decompression pattern (FP16/BF16 +
// convert(FP32)) is transformed to gather node with compressed (FP16/BF16) weights, and decompression convert is moved
// after gather node, so GatherCompressed node should not be generated.
TEST_F(TransformationTestsF, MoveDecompressionAfterGatherFP16Weight) {
    {
        auto input1 = std::make_shared<v0::Parameter>(ov::element::i32, ov::PartialShape{-1, 16});
        auto axis_const = v0::Constant::create(ov::element::i32, ov::Shape{1}, {1});
        auto weights_const = v0::Constant::create(ov::element::f16, ov::Shape{32, 16}, {1});
        auto convert = std::make_shared<v0::Convert>(weights_const, ov::element::f32);
        auto gather = std::make_shared<v8::Gather>(convert, input1, axis_const);

        model = std::make_shared<ov::Model>(ov::OutputVector{gather}, ov::ParameterVector{input1});
        manager.register_pass<MoveDecompressionAfterGather>();
    }
    {
        auto input1 = std::make_shared<v0::Parameter>(ov::element::i32, ov::PartialShape{-1, 16});
        auto axis_const = v0::Constant::create(ov::element::i32, ov::Shape{1}, {1});
        auto weights_const = v0::Constant::create(ov::element::f16, ov::Shape{32, 16}, {1});
        auto gather = std::make_shared<v8::Gather>(weights_const, input1, axis_const);
        auto convert = std::make_shared<v0::Convert>(gather, ov::element::f32);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{convert}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, MoveDecompressionAfterGatherBF16Weight) {
    {
        auto input1 = std::make_shared<v0::Parameter>(ov::element::i32, ov::PartialShape{-1, 16});
        auto axis_const = v0::Constant::create(ov::element::i32, ov::Shape{1}, {1});
        auto weights_const = v0::Constant::create(ov::element::bf16, ov::Shape{32, 16}, {1});
        auto convert = std::make_shared<v0::Convert>(weights_const, ov::element::f32);
        auto gather = std::make_shared<v8::Gather>(convert, input1, axis_const);

        model = std::make_shared<ov::Model>(ov::OutputVector{gather}, ov::ParameterVector{input1});
        manager.register_pass<MoveDecompressionAfterGather>();
    }
    {
        auto input1 = std::make_shared<v0::Parameter>(ov::element::i32, ov::PartialShape{-1, 16});
        auto axis_const = v0::Constant::create(ov::element::i32, ov::Shape{1}, {1});
        auto weights_const = v0::Constant::create(ov::element::bf16, ov::Shape{32, 16}, {1});
        auto gather = std::make_shared<v8::Gather>(weights_const, input1, axis_const);
        auto convert = std::make_shared<v0::Convert>(gather, ov::element::f32);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{convert}, ov::ParameterVector{input1});
    }
}

// LLM weight tying: the Convert(bf16->f32) feeds both a Gather and a MatMul.
// MoveDecompressionAfterGather should still peel the Gather branch off into Gather(bf16) -> Convert(f32),
// leaving the original Convert in place for the MatMul consumer.
TEST_F(TransformationTestsF, MoveDecompressionAfterGatherMultiConsumer) {
    {
        auto indices = std::make_shared<v0::Parameter>(ov::element::i32, ov::PartialShape{-1, 16});
        auto matmul_input = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 16});
        auto axis_const = v0::Constant::create(ov::element::i32, ov::Shape{1}, {1});
        auto weights_const = v0::Constant::create(ov::element::bf16, ov::Shape{32, 16}, {1});
        auto convert = std::make_shared<v0::Convert>(weights_const, ov::element::f32);
        auto gather = std::make_shared<v8::Gather>(convert, indices, axis_const);
        auto matmul = std::make_shared<v0::MatMul>(matmul_input, convert, false, true);

        model =
            std::make_shared<ov::Model>(ov::OutputVector{gather, matmul}, ov::ParameterVector{indices, matmul_input});
        manager.register_pass<MoveDecompressionAfterGather>();
    }
    {
        auto indices = std::make_shared<v0::Parameter>(ov::element::i32, ov::PartialShape{-1, 16});
        auto matmul_input = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 16});
        auto axis_const = v0::Constant::create(ov::element::i32, ov::Shape{1}, {1});
        auto weights_const = v0::Constant::create(ov::element::bf16, ov::Shape{32, 16}, {1});
        auto convert = std::make_shared<v0::Convert>(weights_const, ov::element::f32);
        auto new_gather = std::make_shared<v8::Gather>(weights_const, indices, axis_const);
        auto new_convert = std::make_shared<v0::Convert>(new_gather, ov::element::f32);
        auto matmul = std::make_shared<v0::MatMul>(matmul_input, convert, false, true);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{new_convert, matmul},
                                                ov::ParameterVector{indices, matmul_input});
    }
}
