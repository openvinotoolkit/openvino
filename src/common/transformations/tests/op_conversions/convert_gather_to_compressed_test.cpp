// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_gather_to_compressed.hpp"

#include <memory>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
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

TEST_F(TransformationTestsF, ConvertGatherToCompressed1) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1, 16});
        auto axis_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {1});
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{32, 16}, {1});
        auto convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f32);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{32, 1}, {1});
        auto scale = std::make_shared<ov::op::v1::Multiply>(convert, scale_const);
        auto gather = std::make_shared<ov::op::v8::Gather>(scale, input1, axis_const);

        model = std::make_shared<ov::Model>(ov::NodeVector{gather}, ov::ParameterVector{input1});
        manager.register_pass<ConvertGatherToGatherCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1, 16});
        auto axis_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {1});
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{32, 16}, {1});
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{32, 1}, {1});
        auto gather_compressed =
            std::make_shared<ov::op::internal::GatherCompressed>(weights_const, input1, axis_const, 0, scale_const);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{gather_compressed}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertGatherToCompressed2) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1, 16});
        auto axis_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {1});
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{32, 16}, {1});
        auto convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f32);
        auto zp_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{32, 1}, {1});
        auto sub = std::make_shared<ov::op::v1::Subtract>(convert, zp_const);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{32, 1}, {1});
        auto scale = std::make_shared<ov::op::v1::Multiply>(sub, scale_const);
        auto gather = std::make_shared<ov::op::v8::Gather>(scale, input1, axis_const);

        model = std::make_shared<ov::Model>(ov::NodeVector{gather}, ov::ParameterVector{input1});
        manager.register_pass<ConvertGatherToGatherCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1, 16});
        auto axis_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {1});
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{32, 16}, {1});
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{32, 1}, {1});
        auto zp_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{32, 1}, {1});
        auto gather_compressed = std::make_shared<ov::op::internal::GatherCompressed>(weights_const,
                                                                                      input1,
                                                                                      axis_const,
                                                                                      0,
                                                                                      scale_const,
                                                                                      zp_const);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{gather_compressed}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertGatherToCompressed3) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1, 16});
        auto axis_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {1});
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{32, 4, 4}, {1});
        auto convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f32);
        auto zp_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{32, 4, 1}, {1});
        auto sub = std::make_shared<ov::op::v1::Subtract>(convert, zp_const);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{32, 4, 1}, {1});
        auto scale = std::make_shared<ov::op::v1::Multiply>(sub, scale_const);
        auto reshape_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, {-1, 16});
        auto reshape = std::make_shared<ov::op::v1::Reshape>(scale, reshape_const, false);
        auto gather = std::make_shared<ov::op::v8::Gather>(reshape, input1, axis_const);

        model = std::make_shared<ov::Model>(ov::NodeVector{gather}, ov::ParameterVector{input1});
        manager.register_pass<ConvertGatherToGatherCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1, 16});
        auto axis_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {1});
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{32, 16}, {1});
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{32, 4}, {1});
        auto zp_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{32, 4}, {1});
        auto gather_compressed = std::make_shared<ov::op::internal::GatherCompressed>(weights_const,
                                                                                      input1,
                                                                                      axis_const,
                                                                                      0,
                                                                                      scale_const,
                                                                                      zp_const);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{gather_compressed}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertGatherToCompressed4) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1, 16});
        auto axis_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {1});
        auto weights_const = ov::op::v0::Constant::create(ov::element::u4, ov::Shape{32, 4, 4}, {1});
        auto convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f32);
        auto zp_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 1, 1}, {1});
        auto sub = std::make_shared<ov::op::v1::Subtract>(convert, zp_const);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{32, 4, 1}, {1});
        auto scale = std::make_shared<ov::op::v1::Multiply>(sub, scale_const);
        auto reshape_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, {-1, 16});
        auto reshape = std::make_shared<ov::op::v1::Reshape>(scale, reshape_const, false);
        auto gather = std::make_shared<ov::op::v8::Gather>(reshape, input1, axis_const);

        model = std::make_shared<ov::Model>(ov::NodeVector{gather}, ov::ParameterVector{input1});
        manager.register_pass<ConvertGatherToGatherCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1, 16});
        auto axis_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {1});
        auto weights_const = ov::op::v0::Constant::create(ov::element::u4, ov::Shape{32, 16}, {1});
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{32, 4}, {1});
        auto zp_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 1}, {1});
        auto gather_compressed = std::make_shared<ov::op::internal::GatherCompressed>(weights_const,
                                                                                      input1,
                                                                                      axis_const,
                                                                                      0,
                                                                                      scale_const,
                                                                                      zp_const);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{gather_compressed}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertGatherToCompressedFP16) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1, 16});
        auto axis_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {1});
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{32, 16}, {1});
        auto convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f16);
        auto zp_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{32, 1}, {1});
        auto sub = std::make_shared<ov::op::v1::Subtract>(convert, zp_const);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{32, 1}, {1});
        auto scale = std::make_shared<ov::op::v1::Multiply>(sub, scale_const);
        auto scale_convert = std::make_shared<ov::op::v0::Convert>(scale, ov::element::f32);
        auto gather = std::make_shared<ov::op::v8::Gather>(scale_convert, input1, axis_const);

        model = std::make_shared<ov::Model>(ov::NodeVector{gather}, ov::ParameterVector{input1});
        manager.register_pass<ConvertGatherToGatherCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1, 16});
        auto axis_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {1});
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{32, 16}, {1});
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{32, 1}, {1});
        auto scale_convert = std::make_shared<ov::op::v0::Convert>(scale_const, ov::element::f32);
        auto zp_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{32, 1}, {1});
        auto gather_compressed = std::make_shared<ov::op::internal::GatherCompressed>(weights_const,
                                                                                      input1,
                                                                                      axis_const,
                                                                                      0,
                                                                                      scale_convert,
                                                                                      zp_const);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{gather_compressed}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertGatherToCompressedMultiOutput) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1, 16});

        auto input2 = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{});
        auto topk = std::make_shared<ov::op::v11::TopK>(input1,
                                                        input2,
                                                        0,
                                                        ov::op::v11::TopK::Mode::MAX,
                                                        ov::op::v11::TopK::SortType::SORT_VALUES);

        auto axis_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {1});
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{32, 16}, {1});
        auto convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f32);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{32, 1}, {1});
        auto scale = std::make_shared<ov::op::v1::Multiply>(convert, scale_const);
        auto gather = std::make_shared<ov::op::v8::Gather>(scale, topk->output(1), axis_const);

        model = std::make_shared<ov::Model>(ov::NodeVector{gather}, ov::ParameterVector{input1, input2});
        manager.register_pass<ConvertGatherToGatherCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1, 16});

        auto input2 = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{});
        auto topk = std::make_shared<ov::op::v11::TopK>(input1,
                                                        input2,
                                                        0,
                                                        ov::op::v11::TopK::Mode::MAX,
                                                        ov::op::v11::TopK::SortType::SORT_VALUES);

        auto axis_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {1});
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{32, 16}, {1});
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{32, 1}, {1});
        auto gather_compressed = std::make_shared<ov::op::internal::GatherCompressed>(weights_const,
                                                                                      topk->output(1),
                                                                                      axis_const,
                                                                                      0,
                                                                                      scale_const);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{gather_compressed}, ov::ParameterVector{input1, input2});
    }
}

// In compressed FP16/BF16 weight case, gather node with constant weight decompression pattern (FP16/BF16 +
// convert(FP32)) is transformed to gather node with compressed (FP16/BF16) weights, and decompression convert is moved
// after gather node, so GatherCompressed node should not be generated.
TEST_F(TransformationTestsF, MoveDecompressionAfterGatherFP16Weight) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1, 16});
        auto axis_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {1});
        auto weights_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{32, 16}, {1});
        auto convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f32);
        auto gather = std::make_shared<ov::op::v8::Gather>(convert, input1, axis_const);

        model = std::make_shared<ov::Model>(ov::NodeVector{gather}, ov::ParameterVector{input1});
        manager.register_pass<MoveDecompressionAfterGather>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1, 16});
        auto axis_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {1});
        auto weights_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{32, 16}, {1});
        auto gather = std::make_shared<ov::op::v8::Gather>(weights_const, input1, axis_const);
        auto convert = std::make_shared<ov::op::v0::Convert>(gather, ov::element::f32);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{convert}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, MoveDecompressionAfterGatherBF16Weight) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1, 16});
        auto axis_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {1});
        auto weights_const = ov::op::v0::Constant::create(ov::element::bf16, ov::Shape{32, 16}, {1});
        auto convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f32);
        auto gather = std::make_shared<ov::op::v8::Gather>(convert, input1, axis_const);

        model = std::make_shared<ov::Model>(ov::NodeVector{gather}, ov::ParameterVector{input1});
        manager.register_pass<MoveDecompressionAfterGather>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1, 16});
        auto axis_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {1});
        auto weights_const = ov::op::v0::Constant::create(ov::element::bf16, ov::Shape{32, 16}, {1});
        auto gather = std::make_shared<ov::op::v8::Gather>(weights_const, input1, axis_const);
        auto convert = std::make_shared<ov::op::v0::Convert>(gather, ov::element::f32);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{convert}, ov::ParameterVector{input1});
    }
}
