// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/opsets/opset10.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "transformations/low_precision/mark_dequantization_subgraph.hpp"
#include "transformations/fp16_compression/mark_decompression_convert_constant_folding.hpp"
#include "transformations/rt_info/decompression.hpp"
#include "transformations/rt_info/dequantization_node.hpp"
#include "transformations/rt_info/keep_const_precision.hpp"

#include "common_test_utils/ov_test_utils.hpp"

using namespace ov;

TEST_F(TransformationTestsF, MarkDequantizationSubgraphTransformation) {
    // Input graph:
    //
    //     Parameter
    //      |F32
    //      |
    //     FakeQuantize
    //      |F32
    //      |
    //     Convert   Constant      Constant  Constant
    //      |U8        |U8           |I8       |I8
    //      |          |             |         |
    //     Convert  Convert(DCF) Convert(DCF) Convert(DCF)
    //       \FP32    /FP32          |FP32   /F32
    //        \      /               |      /
    //        Subtract  Constant    Subtract  Constant
    //          \FP32   /FP32        |FP32    /FP32
    //           \     /             |       /
    //           Multiply           Multiply
    //             \FP32            /FP32
    //              \              /
    //                 Convolution
    //
    // After MarkDequantizationSubgraph all Subtract and Multiply nodes from above graph
    // are marked with 'DequantizationNode' attribute.
    // All 'Convert(DCF)' nodes from above graph are marked with 'DisableConstantFolding' attribute
    // Weights and zero points are marked with 'KeepConstPrecision' attribute

    {
        auto parameter = std::make_shared<opset10::Parameter>(element::f32, Shape{1, 16, 14, 14});
        std::shared_ptr<Node> activations =
            std::make_shared<opset10::FakeQuantize>(parameter,
                                                    opset10::Constant::create(element::f32, Shape{}, {0}),
                                                    opset10::Constant::create(element::f32, Shape{}, {20}),
                                                    opset10::Constant::create(element::f32, Shape{}, {0}),
                                                    opset10::Constant::create(element::f32, Shape{}, {254}),
                                                    255);
        {
            auto first_convert = std::make_shared<opset10::Convert>(activations, element::u8);
            auto second_convert = std::make_shared<opset10::Convert>(first_convert, element::f32);
            auto zero_point = opset10::Constant::create(element::u8, Shape{}, {127});
            auto convert_on_zero_point = std::make_shared<opset10::Convert>(zero_point, element::f32);
            auto subtract = std::make_shared<opset10::Subtract>(second_convert, convert_on_zero_point);
            auto scale = opset10::Constant::create(element::f32, Shape{}, {0.2});
            auto multiply = std::make_shared<opset10::Multiply>(subtract, scale);
            activations = multiply;
        }

        std::shared_ptr<Node> weights = opset10::Constant::create(element::i8, Shape{4, 16, 1, 1}, {-2});
        {
            auto convert = std::make_shared<opset10::Convert>(weights, element::f32);
            auto zero_point = opset10::Constant::create(element::i8, Shape{}, {127});
            auto convert_on_zero_point = std::make_shared<opset10::Convert>(zero_point, element::f32);
            auto subtract = std::make_shared<opset10::Subtract>(convert, convert_on_zero_point);
            auto scale = opset10::Constant::create(element::f32, Shape{}, {0.2});
            auto multiply = std::make_shared<opset10::Multiply>(subtract, scale);
            weights = multiply;
        }

        auto conv = std::make_shared<opset10::Convolution>(activations,
                                                           weights,
                                                           Strides{1, 1},
                                                           CoordinateDiff{0, 0},
                                                           CoordinateDiff{0, 0},
                                                           Strides{1, 1});
        model = std::make_shared<Model>(conv, ParameterVector{parameter});
    }

    manager.register_pass<pass::MarkDequantizationSubgraph>(element::TypeVector{element::u8, element::i8});
    manager.register_pass<pass::ConstantFolding>();

    {
        auto parameter = std::make_shared<opset10::Parameter>(element::f32, Shape{1, 16, 14, 14});
        std::shared_ptr<Node> activations =
            std::make_shared<opset10::FakeQuantize>(parameter,
                                                    opset10::Constant::create(element::f32, Shape{}, {0}),
                                                    opset10::Constant::create(element::f32, Shape{}, {20}),
                                                    opset10::Constant::create(element::f32, Shape{}, {0}),
                                                    opset10::Constant::create(element::f32, Shape{}, {254}),
                                                    255);
        {
            auto first_convert = std::make_shared<opset10::Convert>(activations, element::u8);
            auto second_convert = std::make_shared<opset10::Convert>(first_convert, element::f32);
            auto zero_point = opset10::Constant::create(element::u8, Shape{}, {127});
            auto convert_on_zero_point = std::make_shared<opset10::Convert>(zero_point, element::f32);
            pass::disable_constant_folding(convert_on_zero_point);
            auto subtract = std::make_shared<opset10::Subtract>(second_convert, convert_on_zero_point);
            mark_as_dequantization_node(subtract);
            auto scale = opset10::Constant::create(element::f32, Shape{}, {0.2});
            auto multiply = std::make_shared<opset10::Multiply>(subtract, scale);
            mark_as_dequantization_node(multiply);
            activations = multiply;
        }

        std::shared_ptr<Node> weights = opset10::Constant::create(element::i8, Shape{4, 16, 1, 1}, {-2});
        enable_keep_const_precision(weights);
        {
            auto convert = std::make_shared<opset10::Convert>(weights, element::f32);
            pass::disable_constant_folding(convert);
            auto zero_point = opset10::Constant::create(element::i8, Shape{}, {127});
            enable_keep_const_precision(zero_point);
            auto convert_on_zero_point = std::make_shared<opset10::Convert>(zero_point, element::f32);
            pass::disable_constant_folding(convert_on_zero_point);
            auto subtract = std::make_shared<opset10::Subtract>(convert, convert_on_zero_point);
            mark_as_dequantization_node(subtract);
            auto scale = opset10::Constant::create(element::f32, Shape{}, {0.2});
            auto multiply = std::make_shared<opset10::Multiply>(subtract, scale);
            mark_as_dequantization_node(multiply);
            weights = multiply;
        }

        auto conv = std::make_shared<opset10::Convolution>(activations,
                                                           weights,
                                                           Strides{1, 1},
                                                           CoordinateDiff{0, 0},
                                                           CoordinateDiff{0, 0},
                                                           Strides{1, 1});
        model_ref = std::make_shared<Model>(conv, ParameterVector{parameter});
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::RUNTIME_KEYS);
}

TEST_F(TransformationTestsF, MarkDequantizationSubgraphTransformationNoZeroPoint) {
    // Input graph:
    //
    //     Parameter
    //      |F32
    //      |
    //     FakeQuantize
    //      |F32
    //      |
    //     Convert             Constant
    //      |U8                   |I8
    //      |                     |
    //     Convert  Constant   Convert(DCF) Constant
    //       \FP32    /FP32       \FP32    /F32
    //        \      /             \      /
    //         Multiply            Multiply
    //             \FP32            /FP32
    //              \              /
    //                 Convolution
    //
    // After MarkDequantizationSubgraph all Multiply nodes from above graph
    // are marked with 'DequantizationNode' attribute.
    // Also 'Convert(DCF)' node from above graph is marked with 'DisableConstantFolding' attribute
    // Weights node is marked with 'KeepConstPrecision' attribute

    {
        auto parameter = std::make_shared<opset10::Parameter>(element::f32, Shape{1, 16, 14, 14});
        std::shared_ptr<Node> activations =
            std::make_shared<opset10::FakeQuantize>(parameter,
                                                    opset10::Constant::create(element::f32, Shape{}, {0}),
                                                    opset10::Constant::create(element::f32, Shape{}, {20}),
                                                    opset10::Constant::create(element::f32, Shape{}, {0}),
                                                    opset10::Constant::create(element::f32, Shape{}, {254}),
                                                    255);
        {
            auto first_convert = std::make_shared<opset10::Convert>(activations, element::u8);
            auto second_convert = std::make_shared<opset10::Convert>(first_convert, element::f32);
            auto scale = opset10::Constant::create(element::f32, Shape{}, {0.2});
            auto multiply = std::make_shared<opset10::Multiply>(second_convert, scale);
            activations = multiply;
        }

        std::shared_ptr<Node> weights = opset10::Constant::create(element::i8, Shape{4, 16, 1, 1}, {-2});
        {
            auto convert = std::make_shared<opset10::Convert>(weights, element::f32);
            auto scale = opset10::Constant::create(element::f32, Shape{}, {0.2});
            auto multiply = std::make_shared<opset10::Multiply>(convert, scale);
            weights = multiply;
        }

        auto conv = std::make_shared<opset10::Convolution>(activations,
                                                           weights,
                                                           Strides{1, 1},
                                                           CoordinateDiff{0, 0},
                                                           CoordinateDiff{0, 0},
                                                           Strides{1, 1});
        model = std::make_shared<Model>(conv, ParameterVector{parameter});
    }

    manager.register_pass<pass::MarkDequantizationSubgraph>(element::TypeVector{element::u8, element::i8});
    manager.register_pass<pass::ConstantFolding>();

    {
        auto parameter = std::make_shared<opset10::Parameter>(element::f32, Shape{1, 16, 14, 14});
        std::shared_ptr<Node> activations =
            std::make_shared<opset10::FakeQuantize>(parameter,
                                                    opset10::Constant::create(element::f32, Shape{}, {0}),
                                                    opset10::Constant::create(element::f32, Shape{}, {20}),
                                                    opset10::Constant::create(element::f32, Shape{}, {0}),
                                                    opset10::Constant::create(element::f32, Shape{}, {254}),
                                                    255);
        {
            auto first_convert = std::make_shared<opset10::Convert>(activations, element::u8);
            auto second_convert = std::make_shared<opset10::Convert>(first_convert, element::f32);
            auto scale = opset10::Constant::create(element::f32, Shape{}, {0.2});
            auto multiply = std::make_shared<opset10::Multiply>(second_convert, scale);
            mark_as_dequantization_node(multiply);
            activations = multiply;
        }

        std::shared_ptr<Node> weights = opset10::Constant::create(element::i8, Shape{4, 16, 1, 1}, {-2});
        enable_keep_const_precision(weights);
        {
            auto convert = std::make_shared<opset10::Convert>(weights, element::f32);
            pass::disable_constant_folding(convert);
            auto scale = opset10::Constant::create(element::f32, Shape{}, {0.2});
            auto multiply = std::make_shared<opset10::Multiply>(convert, scale);
            mark_as_dequantization_node(multiply);
            weights = multiply;
        }

        auto conv = std::make_shared<opset10::Convolution>(activations,
                                                           weights,
                                                           Strides{1, 1},
                                                           CoordinateDiff{0, 0},
                                                           CoordinateDiff{0, 0},
                                                           Strides{1, 1});
        model_ref = std::make_shared<Model>(conv, ParameterVector{parameter});
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::RUNTIME_KEYS);
}

TEST_F(TransformationTestsF, MarkDequantizationSubgraphTransformationNoZeroPointFP16) {
    // Input graph:
    //
    //     Parameter
    //      |F32
    //      |
    //     FakeQuantize
    //      |F32
    //      |
    //     Convert  Constant   Constant        Constant
    //      |U8       |FP16       |I8         /FP16
    //      |         |           |          /
    //     Convert  Convert    Convert(DCF) Convert
    //       \FP32    /FP32       \FP32    /F32
    //        \      /             \      /
    //         Multiply            Multiply
    //             \FP32            /FP32
    //              \              /
    //                 Convolution
    //
    // After MarkDequantizationSubgraph all Multiply nodes from above graph
    // are marked with 'DequantizationNode' attribute.
    // Also 'Convert(DCF)' node from above graph is marked with 'DisableConstantFolding' attribute
    // Weights node is marked with 'KeepConstPrecision' attribute

    {
        auto parameter = std::make_shared<opset10::Parameter>(element::f32, Shape{1, 16, 14, 14});
        std::shared_ptr<Node> activations =
            std::make_shared<opset10::FakeQuantize>(parameter,
                                                    opset10::Constant::create(element::f32, Shape{}, {0}),
                                                    opset10::Constant::create(element::f32, Shape{}, {20}),
                                                    opset10::Constant::create(element::f32, Shape{}, {0}),
                                                    opset10::Constant::create(element::f32, Shape{}, {254}),
                                                    255);
        {
            auto first_convert = std::make_shared<opset10::Convert>(activations, element::u8);
            auto second_convert = std::make_shared<opset10::Convert>(first_convert, element::f32);
            auto scale = opset10::Constant::create(element::f16, Shape{}, {0.2});
            auto scale_convert = std::make_shared<opset10::Convert>(scale, element::f32);
            mark_as_decompression(scale_convert);
            auto multiply = std::make_shared<opset10::Multiply>(second_convert, scale_convert);
            activations = multiply;
        }

        std::shared_ptr<Node> weights = opset10::Constant::create(element::i8, Shape{4, 16, 1, 1}, {-2});
        {
            auto convert = std::make_shared<opset10::Convert>(weights, element::f32);
            auto scale = opset10::Constant::create(element::f16, Shape{}, {0.2});
            auto scale_convert = std::make_shared<opset10::Convert>(scale, element::f32);
            mark_as_decompression(scale_convert);
            auto multiply = std::make_shared<opset10::Multiply>(convert, scale_convert);
            weights = multiply;
        }

        auto conv = std::make_shared<opset10::Convolution>(activations,
                                                           weights,
                                                           Strides{1, 1},
                                                           CoordinateDiff{0, 0},
                                                           CoordinateDiff{0, 0},
                                                           Strides{1, 1});
        model = std::make_shared<Model>(conv, ParameterVector{parameter});
    }

    manager.register_pass<pass::MarkDequantizationSubgraph>(element::TypeVector{element::u8, element::i8});
    manager.register_pass<pass::DisableDecompressionConvertConstantFolding>();
    manager.register_pass<pass::ConstantFolding>();

    {
        auto parameter = std::make_shared<opset10::Parameter>(element::f32, Shape{1, 16, 14, 14});
        std::shared_ptr<Node> activations =
            std::make_shared<opset10::FakeQuantize>(parameter,
                                                    opset10::Constant::create(element::f32, Shape{}, {0}),
                                                    opset10::Constant::create(element::f32, Shape{}, {20}),
                                                    opset10::Constant::create(element::f32, Shape{}, {0}),
                                                    opset10::Constant::create(element::f32, Shape{}, {254}),
                                                    255);
        {
            auto first_convert = std::make_shared<opset10::Convert>(activations, element::u8);
            auto second_convert = std::make_shared<opset10::Convert>(first_convert, element::f32);
            auto scale = opset10::Constant::create(element::f16, Shape{}, {0.2});
            auto scale_convert = std::make_shared<opset10::Convert>(scale, element::f32);
            mark_as_decompression(scale_convert);
            auto multiply = std::make_shared<opset10::Multiply>(second_convert, scale_convert);
            mark_as_dequantization_node(multiply);
            activations = multiply;
        }

        std::shared_ptr<Node> weights = opset10::Constant::create(element::i8, Shape{4, 16, 1, 1}, {-2});
        enable_keep_const_precision(weights);
        {
            auto convert = std::make_shared<opset10::Convert>(weights, element::f32);
            pass::disable_constant_folding(convert);
            auto scale = opset10::Constant::create(element::f16, Shape{}, {0.2});
            auto scale_convert = std::make_shared<opset10::Convert>(scale, element::f32);
            mark_as_decompression(scale_convert);
            auto multiply = std::make_shared<opset10::Multiply>(convert, scale_convert);
            mark_as_dequantization_node(multiply);
            weights = multiply;
        }

        auto conv = std::make_shared<opset10::Convolution>(activations,
                                                           weights,
                                                           Strides{1, 1},
                                                           CoordinateDiff{0, 0},
                                                           CoordinateDiff{0, 0},
                                                           Strides{1, 1});
        model_ref = std::make_shared<Model>(conv, ParameterVector{parameter});
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::RUNTIME_KEYS);
}

TEST_F(TransformationTestsF, MarkDequantizationSubgraphTransformationNotConstantWeights) {
    // Input graph:
    //
    //     Parameter
    //      |F32
    //      |
    //     FakeQuantize            Constant
    //      |F32                     |I8
    //      |                        |
    //     Convert   Constant       Clamp    Constant
    //      |U8        |U8           |I8       |I8
    //      |          |             |         |
    //     Convert  Convert(DCF) Convert(DCF) Convert(DCF)
    //       \FP32    /FP32          |FP32   /F32
    //        \      /               |      /
    //        Subtract  Constant    Subtract  Constant
    //          \FP32   /FP32        |FP32    /FP32
    //           \     /             |       /
    //           Multiply           Multiply
    //             \FP32            /FP32
    //              \              /
    //                 Convolution
    //
    // After MarkDequantizationSubgraph all Subtract and Multiply nodes from above graph
    // are marked with 'DequantizationNode' attribute.
    // Also all 'Convert(DCF)' nodes from above graph are marked with 'DisableConstantFolding' attribute
    // Weights and zero point nodes are marked with 'KeepConstPrecision' attribute

    {
        auto parameter = std::make_shared<opset10::Parameter>(element::f32, Shape{1, 16, 14, 14});
        std::shared_ptr<Node> activations =
            std::make_shared<opset10::FakeQuantize>(parameter,
                                                    opset10::Constant::create(element::f32, Shape{}, {0}),
                                                    opset10::Constant::create(element::f32, Shape{}, {20}),
                                                    opset10::Constant::create(element::f32, Shape{}, {0}),
                                                    opset10::Constant::create(element::f32, Shape{}, {254}),
                                                    255);
        {
            auto first_convert = std::make_shared<opset10::Convert>(activations, element::u8);
            auto second_convert = std::make_shared<opset10::Convert>(first_convert, element::f32);
            auto zero_point = opset10::Constant::create(element::u8, Shape{}, {127});
            auto convert_on_zero_point = std::make_shared<opset10::Convert>(zero_point, element::f32);
            auto subtract = std::make_shared<opset10::Subtract>(second_convert, convert_on_zero_point);
            auto scale = opset10::Constant::create(element::f32, Shape{}, {0.2});
            auto multiply = std::make_shared<opset10::Multiply>(subtract, scale);
            activations = multiply;
        }

        std::shared_ptr<Node> weights = opset10::Constant::create(element::i8, Shape{4, 16, 1, 1}, {-3});
        enable_keep_const_precision(weights);
        {
            auto clamp = std::make_shared<opset10::Clamp>(weights, -2, 2);
            auto convert = std::make_shared<opset10::Convert>(clamp, element::f32);
            auto zero_point = opset10::Constant::create(element::i8, Shape{}, {127});
            enable_keep_const_precision(zero_point);
            auto convert_on_zero_point = std::make_shared<opset10::Convert>(zero_point, element::f32);
            auto subtract = std::make_shared<opset10::Subtract>(convert, convert_on_zero_point);
            auto scale = opset10::Constant::create(element::f32, Shape{}, {0.2});
            auto multiply = std::make_shared<opset10::Multiply>(subtract, scale);
            weights = multiply;
        }

        auto conv = std::make_shared<opset10::Convolution>(activations,
                                                           weights,
                                                           Strides{1, 1},
                                                           CoordinateDiff{0, 0},
                                                           CoordinateDiff{0, 0},
                                                           Strides{1, 1});
        model = std::make_shared<Model>(conv, ParameterVector{parameter});
    }

    manager.register_pass<pass::MarkDequantizationSubgraph>(element::TypeVector{element::u8, element::i8});
    manager.register_pass<pass::ConstantFolding>();

    {
        auto parameter = std::make_shared<opset10::Parameter>(element::f32, Shape{1, 16, 14, 14});
        std::shared_ptr<Node> activations =
            std::make_shared<opset10::FakeQuantize>(parameter,
                                                    opset10::Constant::create(element::f32, Shape{}, {0}),
                                                    opset10::Constant::create(element::f32, Shape{}, {20}),
                                                    opset10::Constant::create(element::f32, Shape{}, {0}),
                                                    opset10::Constant::create(element::f32, Shape{}, {254}),
                                                    255);
        {
            auto first_convert = std::make_shared<opset10::Convert>(activations, element::u8);
            auto second_convert = std::make_shared<opset10::Convert>(first_convert, element::f32);
            auto zero_point = opset10::Constant::create(element::u8, Shape{}, {127});
            auto convert_on_zero_point = std::make_shared<opset10::Convert>(zero_point, element::f32);
            pass::disable_constant_folding(convert_on_zero_point);
            auto subtract = std::make_shared<opset10::Subtract>(second_convert, convert_on_zero_point);
            mark_as_dequantization_node(subtract);
            auto scale = opset10::Constant::create(element::f32, Shape{}, {0.2});
            auto multiply = std::make_shared<opset10::Multiply>(subtract, scale);
            mark_as_dequantization_node(multiply);
            activations = multiply;
        }

        std::shared_ptr<Node> weights = opset10::Constant::create(element::i8, Shape{4, 16, 1, 1}, {-2});
        {
            // Clamp was constantfolded
            auto convert = std::make_shared<opset10::Convert>(weights, element::f32);
            pass::disable_constant_folding(convert);
            auto zero_point = opset10::Constant::create(element::i8, Shape{}, {127});
            auto convert_on_zero_point = std::make_shared<opset10::Convert>(zero_point, element::f32);
            pass::disable_constant_folding(convert_on_zero_point);
            auto subtract = std::make_shared<opset10::Subtract>(convert, convert_on_zero_point);
            mark_as_dequantization_node(subtract);
            auto scale = opset10::Constant::create(element::f32, Shape{}, {0.2});
            auto multiply = std::make_shared<opset10::Multiply>(subtract, scale);
            mark_as_dequantization_node(multiply);
            weights = multiply;
        }

        auto conv = std::make_shared<opset10::Convolution>(activations,
                                                           weights,
                                                           Strides{1, 1},
                                                           CoordinateDiff{0, 0},
                                                           CoordinateDiff{0, 0},
                                                           Strides{1, 1});
        model_ref = std::make_shared<Model>(conv, ParameterVector{parameter});
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::RUNTIME_KEYS);
}

TEST_F(TransformationTestsF, MarkDequantizationSubgraphTransformationFoldSubConst) {
    // Input graph:           After transformation:
    //
    // Constant    Constant               Constant
    //    |U8         |U8                    |U8
    //    |           |                      |
    // Convert     Convert                Convert(DCF)  Constant
    //    |FP32   /F32                       |FP32      /FP32
    //    |      /                            \        /
    //   Subtract  Constant                    Subtract   Constant
    //    |FP32    /FP32                          |FP32     /FP32
    //    |       /                                \        /
    //   Multiply                                   Multiply
    //
    // After MarkDequantizationSubgraph all Subtract and Multiply nodes from above graph
    // are marked with 'DequantizationNode' attribute.
    // Also all 'Convert(DCF)' node before weights is marked with 'DisableConstantFolding' attribute
    // but Convert before Dequantization Sub const isn't because fold_subtract_const is set to true
    // Weights node is marked with 'KeepConstPrecision' attribute

    {
        auto weights = opset10::Constant::create(element::u8, Shape{4, 16, 1, 1}, {3});
        auto convert = std::make_shared<opset10::Convert>(weights, element::f32);
        auto zero_point = opset10::Constant::create(element::u8, Shape{}, {127});
        auto convert_on_zero_point = std::make_shared<opset10::Convert>(zero_point, element::f32);
        auto subtract = std::make_shared<opset10::Subtract>(convert, convert_on_zero_point);
        auto scale = opset10::Constant::create(element::f32, Shape{}, {0.2});
        auto multiply = std::make_shared<opset10::Multiply>(subtract, scale);
        model = std::make_shared<ov::Model>(ov::OutputVector{multiply});
    }

    manager.register_pass<pass::MarkDequantizationSubgraph>(element::TypeVector{element::u8}, true);
    manager.register_pass<pass::ConstantFolding>();

    {
        auto weights = opset10::Constant::create(element::u8, Shape{4, 16, 1, 1}, {3});
        enable_keep_const_precision(weights);
        auto convert = std::make_shared<opset10::Convert>(weights, element::f32);
        pass::disable_constant_folding(convert);
        auto zero_point = opset10::Constant::create(element::f32, Shape{}, {127});
        auto subtract = std::make_shared<opset10::Subtract>(convert, zero_point);
        mark_as_dequantization_node(subtract);
        auto scale = opset10::Constant::create(element::f32, Shape{}, {0.2});
        auto multiply = std::make_shared<opset10::Multiply>(subtract, scale);
        mark_as_dequantization_node(multiply);
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{multiply});
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::RUNTIME_KEYS);
}
