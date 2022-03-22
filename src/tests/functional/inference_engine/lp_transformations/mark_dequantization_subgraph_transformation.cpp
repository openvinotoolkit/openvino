// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset8.hpp>

#include <transformations/low_precision/mark_dequantization_subgraph.hpp>
#include <transformations/rt_info/dequantization_node.hpp>
#include <ngraph/pass/constant_folding.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"


using namespace ngraph;

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
    // Also all 'Convert(DCF)' nodes from above graph are marked with 'DisableConstantFolding' attribute

    {
        auto parameter = std::make_shared<opset8::Parameter>(element::f32, Shape{1, 16, 14, 14});
        std::shared_ptr<Node> activations = std::make_shared<opset8::FakeQuantize>(parameter,
                opset8::Constant::create(element::f32, Shape{}, {0}),
                opset8::Constant::create(element::f32, Shape{}, {20}),
                opset8::Constant::create(element::f32, Shape{}, {0}),
                opset8::Constant::create(element::f32, Shape{}, {254}), 255);
        {
            auto first_convert = std::make_shared<opset8::Convert>(activations, element::u8);
            auto second_convert = std::make_shared<opset8::Convert>(first_convert, element::f32);
            auto zero_point = opset8::Constant::create(element::u8, Shape{}, {127});
            auto convert_on_zero_point = std::make_shared<opset8::Convert>(zero_point, element::f32);
            auto subtract = std::make_shared<opset8::Subtract>(second_convert, convert_on_zero_point);
            auto scale = opset8::Constant::create(element::f32, Shape{}, {0.2});
            auto multiply = std::make_shared<opset8::Multiply>(subtract, scale);
            activations = multiply;
        }

        std::shared_ptr<Node> weights = opset8::Constant::create(element::i8, Shape{4, 16, 1, 1}, {-2});
        {
            auto convert = std::make_shared<opset8::Convert>(weights, element::f32);
            auto zero_point = opset8::Constant::create(element::i8, Shape{}, {127});
            auto convert_on_zero_point = std::make_shared<opset8::Convert>(zero_point, element::f32);
            auto subtract = std::make_shared<opset8::Subtract>(convert, convert_on_zero_point);
            auto scale = opset8::Constant::create(element::f32, Shape{}, {0.2});
            auto multiply = std::make_shared<opset8::Multiply>(subtract, scale);
            weights = multiply;
        }

        auto conv = std::make_shared<opset8::Convolution>(activations, weights, Strides{1, 1}, CoordinateDiff{0, 0}, CoordinateDiff{0, 0}, Strides{1, 1});
        function = std::make_shared<ngraph::Function>(conv, ParameterVector{parameter});
    }

    manager.register_pass<ngraph::pass::MarkDequantizationSubgraph>();
    manager.register_pass<ov::pass::ConstantFolding>();

    {
        auto parameter = std::make_shared<opset8::Parameter>(element::f32, Shape{1, 16, 14, 14});
        std::shared_ptr<Node> activations = std::make_shared<opset8::FakeQuantize>(parameter,
                opset8::Constant::create(element::f32, Shape{}, {0}),
                opset8::Constant::create(element::f32, Shape{}, {20}),
                opset8::Constant::create(element::f32, Shape{}, {0}),
                opset8::Constant::create(element::f32, Shape{}, {254}), 255);
        {
            auto first_convert = std::make_shared<opset8::Convert>(activations, element::u8);
            auto second_convert = std::make_shared<opset8::Convert>(first_convert, element::f32);
            auto zero_point = opset8::Constant::create(element::u8, Shape{}, {127});
            auto convert_on_zero_point = std::make_shared<opset8::Convert>(zero_point, element::f32);
            ov::pass::disable_constant_folding(convert_on_zero_point);
            auto subtract = std::make_shared<opset8::Subtract>(second_convert, convert_on_zero_point);
            ov::mark_as_dequantization_node(subtract);
            auto scale = opset8::Constant::create(element::f32, Shape{}, {0.2});
            auto multiply = std::make_shared<opset8::Multiply>(subtract, scale);
            ov::mark_as_dequantization_node(multiply);
            activations = multiply;
        }

        std::shared_ptr<Node> weights = opset8::Constant::create(element::i8, Shape{4, 16, 1, 1}, {-2});
        {
            auto convert = std::make_shared<opset8::Convert>(weights, element::f32);
            ov::pass::disable_constant_folding(convert);
            auto zero_point = opset8::Constant::create(element::i8, Shape{}, {127});
            auto convert_on_zero_point = std::make_shared<opset8::Convert>(zero_point, element::f32);
            ov::pass::disable_constant_folding(convert_on_zero_point);
            auto subtract = std::make_shared<opset8::Subtract>(convert, convert_on_zero_point);
            ov::mark_as_dequantization_node(subtract);
            auto scale = opset8::Constant::create(element::f32, Shape{}, {0.2});
            auto multiply = std::make_shared<opset8::Multiply>(subtract, scale);
            ov::mark_as_dequantization_node(multiply);
            weights = multiply;
        }

        auto conv = std::make_shared<opset8::Convolution>(activations, weights, Strides{1, 1}, CoordinateDiff{0, 0}, CoordinateDiff{0, 0}, Strides{1, 1});
        function_ref = std::make_shared<ngraph::Function>(conv, ParameterVector{parameter});
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

    {
        auto parameter = std::make_shared<opset8::Parameter>(element::f32, Shape{1, 16, 14, 14});
        std::shared_ptr<Node> activations = std::make_shared<opset8::FakeQuantize>(parameter,
                opset8::Constant::create(element::f32, Shape{}, {0}),
                opset8::Constant::create(element::f32, Shape{}, {20}),
                opset8::Constant::create(element::f32, Shape{}, {0}),
                opset8::Constant::create(element::f32, Shape{}, {254}), 255);
        {
            auto first_convert = std::make_shared<opset8::Convert>(activations, element::u8);
            auto second_convert = std::make_shared<opset8::Convert>(first_convert, element::f32);
            auto scale = opset8::Constant::create(element::f32, Shape{}, {0.2});
            auto multiply = std::make_shared<opset8::Multiply>(second_convert, scale);
            activations = multiply;
        }

        std::shared_ptr<Node> weights = opset8::Constant::create(element::i8, Shape{4, 16, 1, 1}, {-2});
        {
            auto convert = std::make_shared<opset8::Convert>(weights, element::f32);
            auto scale = opset8::Constant::create(element::f32, Shape{}, {0.2});
            auto multiply = std::make_shared<opset8::Multiply>(convert, scale);
            weights = multiply;
        }

        auto conv = std::make_shared<opset8::Convolution>(activations, weights, Strides{1, 1}, CoordinateDiff{0, 0}, CoordinateDiff{0, 0}, Strides{1, 1});
        function = std::make_shared<ngraph::Function>(conv, ParameterVector{parameter});
    }

    manager.register_pass<ngraph::pass::MarkDequantizationSubgraph>();
    manager.register_pass<ov::pass::ConstantFolding>();

    {
        auto parameter = std::make_shared<opset8::Parameter>(element::f32, Shape{1, 16, 14, 14});
        std::shared_ptr<Node> activations = std::make_shared<opset8::FakeQuantize>(parameter,
                opset8::Constant::create(element::f32, Shape{}, {0}),
                opset8::Constant::create(element::f32, Shape{}, {20}),
                opset8::Constant::create(element::f32, Shape{}, {0}),
                opset8::Constant::create(element::f32, Shape{}, {254}), 255);
        {
            auto first_convert = std::make_shared<opset8::Convert>(activations, element::u8);
            auto second_convert = std::make_shared<opset8::Convert>(first_convert, element::f32);
            auto scale = opset8::Constant::create(element::f32, Shape{}, {0.2});
            auto multiply = std::make_shared<opset8::Multiply>(second_convert, scale);
            ov::mark_as_dequantization_node(multiply);
            activations = multiply;
        }

        std::shared_ptr<Node> weights = opset8::Constant::create(element::i8, Shape{4, 16, 1, 1}, {-2});
        {
            auto convert = std::make_shared<opset8::Convert>(weights, element::f32);
            ov::pass::disable_constant_folding(convert);
            auto scale = opset8::Constant::create(element::f32, Shape{}, {0.2});
            auto multiply = std::make_shared<opset8::Multiply>(convert, scale);
            ov::mark_as_dequantization_node(multiply);
            weights = multiply;
        }

        auto conv = std::make_shared<opset8::Convolution>(activations, weights, Strides{1, 1}, CoordinateDiff{0, 0}, CoordinateDiff{0, 0}, Strides{1, 1});
        function_ref = std::make_shared<ngraph::Function>(conv, ParameterVector{parameter});
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::RUNTIME_KEYS);
}
