// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/convolution_to_group_convolution_fusion.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/opsets/opset10_decl.hpp"
#include "openvino/pass/manager.hpp"

using namespace testing;
using namespace ov;

TEST_F(TransformationTestsF, ConvToGroupConvFusionSplit) {
    Shape input_shape{2, 10, 14, 14};
    size_t num_splits = 5;
    int axis = 1;
    Shape weights_shape{3, input_shape[1] / num_splits, 1, 1};
    const auto spatial_dim_size = weights_shape.size() - 2;
    Strides strides(spatial_dim_size, 1);
    CoordinateDiff pads_begin(spatial_dim_size, 0);
    CoordinateDiff pads_end(spatial_dim_size, 0);
    Strides dilations(spatial_dim_size, 1);

    {
        const auto data = std::make_shared<opset10::Parameter>(element::f32, input_shape);
        const auto axis_node = opset10::Constant::create(element::i32, Shape{}, {axis});
        const auto split = std::make_shared<opset10::Split>(data, axis_node, num_splits);
        OutputVector concat_inputs;
        concat_inputs.reserve(num_splits);
        for (size_t i = 0; i < num_splits; i++) {
            const auto weights = opset10::Constant::create(element::f32, weights_shape, {i + 1});
            concat_inputs.push_back(std::make_shared<opset10::Convolution>(split->output(i),
                                                                           weights,
                                                                           strides,
                                                                           pads_begin,
                                                                           pads_end,
                                                                           dilations));
        }
        const auto concat = std::make_shared<opset10::Concat>(concat_inputs, axis);
        model = std::make_shared<Model>(concat, ParameterVector{data});
        manager.register_pass<ov::pass::ConvolutionToGroupConvolutionFusion>();
    }

    {
        const auto data = std::make_shared<opset10::Parameter>(element::f32, input_shape);
        OutputVector concat_inputs;
        concat_inputs.reserve(num_splits);
        Shape new_weights_shape = weights_shape;
        new_weights_shape.insert(new_weights_shape.begin(), 1);
        for (size_t i = 0; i < num_splits; i++) {
            const auto weights = opset10::Constant::create(element::f32, new_weights_shape, {i + 1});
            concat_inputs.push_back(weights);
        }
        const auto concat = std::make_shared<opset10::Concat>(concat_inputs, 0);
        const auto conv =
            std::make_shared<opset10::GroupConvolution>(data, concat, strides, pads_begin, pads_end, dilations);
        model_ref = std::make_shared<Model>(conv, ParameterVector{data});
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, ConvToGroupConvFusionVariadicSplit) {
    Shape input_shape{2, 10, 14, 14};
    size_t num_splits = 5;
    int axis = 1;
    Shape weights_shape{3, input_shape[1] / num_splits, 1, 1};
    const auto spatial_dim_size = weights_shape.size() - 2;
    Strides strides(spatial_dim_size, 1);
    CoordinateDiff pads_begin(spatial_dim_size, 0);
    CoordinateDiff pads_end(spatial_dim_size, 0);
    Strides dilations(spatial_dim_size, 1);

    {
        const auto data = std::make_shared<opset10::Parameter>(element::f32, input_shape);
        const auto axis_node = opset10::Constant::create(element::i32, Shape{}, {axis});
        const auto split_lengths =
            opset10::Constant::create(element::i32, Shape{num_splits}, std::vector<int>(num_splits, 2));
        const auto split = std::make_shared<opset10::VariadicSplit>(data, axis_node, split_lengths);
        OutputVector concat_inputs;
        concat_inputs.reserve(num_splits);
        for (size_t i = 0; i < num_splits; i++) {
            const auto weights = opset10::Constant::create(element::f32, weights_shape, {i + 1});
            concat_inputs.push_back(std::make_shared<opset10::Convolution>(split->output(i),
                                                                           weights,
                                                                           strides,
                                                                           pads_begin,
                                                                           pads_end,
                                                                           dilations));
        }
        const auto concat = std::make_shared<opset10::Concat>(concat_inputs, axis);
        model = std::make_shared<Model>(concat, ParameterVector{data});
        manager.register_pass<ov::pass::ConvolutionToGroupConvolutionFusion>();
    }

    {
        const auto data = std::make_shared<opset10::Parameter>(element::f32, input_shape);
        OutputVector concat_inputs;
        concat_inputs.reserve(num_splits);
        Shape new_weights_shape = weights_shape;
        new_weights_shape.insert(new_weights_shape.begin(), 1);
        for (size_t i = 0; i < num_splits; i++) {
            const auto weights = opset10::Constant::create(element::f32, new_weights_shape, {i + 1});
            concat_inputs.push_back(weights);
        }
        const auto concat = std::make_shared<opset10::Concat>(concat_inputs, 0);
        const auto conv =
            std::make_shared<opset10::GroupConvolution>(data, concat, strides, pads_begin, pads_end, dilations);
        model_ref = std::make_shared<Model>(conv, ParameterVector{data});
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, NegativeConvToGroupConvFusionSplitInvalidAxis) {
    Shape input_shape{2, 10, 14, 14};
    int num_splits = 2;
    int axis = 2;
    Shape weights_shape{3, input_shape[1], 1, 1};
    const auto spatial_dim_size = weights_shape.size() - 2;
    Strides strides(spatial_dim_size, 1);
    CoordinateDiff pads_begin(spatial_dim_size, 0);
    CoordinateDiff pads_end(spatial_dim_size, 0);
    Strides dilations(spatial_dim_size, 1);

    {
        const auto data = std::make_shared<opset10::Parameter>(element::f32, input_shape);
        const auto axis_node = opset10::Constant::create(element::i32, Shape{}, {axis});
        const auto split = std::make_shared<opset10::Split>(data, axis_node, num_splits);
        OutputVector concat_inputs;
        concat_inputs.reserve(num_splits);
        for (int i = 0; i < num_splits; i++) {
            const auto weights = opset10::Constant::create(element::f32, weights_shape, {i + 1});
            concat_inputs.push_back(std::make_shared<opset10::Convolution>(split->output(i),
                                                                           weights,
                                                                           strides,
                                                                           pads_begin,
                                                                           pads_end,
                                                                           dilations));
        }
        const auto concat = std::make_shared<opset10::Concat>(concat_inputs, axis);
        model = std::make_shared<Model>(concat, ParameterVector{data});
        manager.register_pass<ov::pass::ConvolutionToGroupConvolutionFusion>();
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, NegativeConvToGroupConvFusionSplitNotMatchingConvAttributes) {
    Shape input_shape{2, 10, 14, 14};
    size_t num_splits = 2;
    int axis = 1;
    const auto spatial_dim_size = 2;

    {
        const auto data = std::make_shared<opset10::Parameter>(element::f32, input_shape);
        const auto axis_node = opset10::Constant::create(element::i32, Shape{}, {axis});
        const auto split = std::make_shared<opset10::Split>(data, axis_node, num_splits);

        const auto weights1 = opset10::Constant::create(element::f32, Shape{3, input_shape[1] / num_splits, 2, 2}, {1});
        Strides strides1(spatial_dim_size, 1);
        CoordinateDiff pads_begin1(spatial_dim_size, 1);
        CoordinateDiff pads_end1(spatial_dim_size, 1);
        Strides dilations1(spatial_dim_size, 1);
        const auto conv1 = std::make_shared<opset10::Convolution>(split->output(0),
                                                                  weights1,
                                                                  strides1,
                                                                  pads_begin1,
                                                                  pads_end1,
                                                                  dilations1);

        const auto weights2 = opset10::Constant::create(element::f32, Shape{3, input_shape[1] / num_splits, 4, 4}, {1});
        Strides strides2(spatial_dim_size, 1);
        CoordinateDiff pads_begin2(spatial_dim_size, 2);
        CoordinateDiff pads_end2(spatial_dim_size, 2);
        Strides dilations2(spatial_dim_size, 1);
        const auto conv2 = std::make_shared<opset10::Convolution>(split->output(1),
                                                                  weights2,
                                                                  strides2,
                                                                  pads_begin2,
                                                                  pads_end2,
                                                                  dilations2);
        const auto concat = std::make_shared<opset10::Concat>(OutputVector{conv1, conv2}, axis);
        model = std::make_shared<Model>(concat, ParameterVector{data});
        manager.register_pass<ov::pass::ConvolutionToGroupConvolutionFusion>();
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, NegativeConvToGroupConvFusionVariadicSplitUnevenSplitLengths) {
    Shape input_shape{2, 10, 14, 14};
    int axis = 1;
    const auto spatial_dim_size = 2;
    Strides strides(spatial_dim_size, 1);
    CoordinateDiff pads_begin(spatial_dim_size, 0);
    CoordinateDiff pads_end(spatial_dim_size, 0);
    Strides dilations(spatial_dim_size, 1);

    {
        const auto data = std::make_shared<opset10::Parameter>(element::f32, input_shape);
        const auto axis_node = opset10::Constant::create(element::i32, Shape{}, {axis});
        const auto split_lengths = opset10::Constant::create(element::i32,
                                                             Shape{2},
                                                             std::vector<int>{3, static_cast<int>(input_shape[1]) - 3});
        const auto split = std::make_shared<opset10::VariadicSplit>(data, axis_node, split_lengths);
        const auto weights1 = opset10::Constant::create(element::f32, Shape{3, 3, 1, 1}, {1});
        const auto conv1 = std::make_shared<opset10::Convolution>(split->output(0),
                                                                  weights1,
                                                                  strides,
                                                                  pads_begin,
                                                                  pads_end,
                                                                  dilations);
        const auto weights2 = opset10::Constant::create(element::f32, Shape{3, 7, 1, 1}, {2});
        const auto conv2 = std::make_shared<opset10::Convolution>(split->output(1),
                                                                  weights2,
                                                                  strides,
                                                                  pads_begin,
                                                                  pads_end,
                                                                  dilations);
        const auto concat = std::make_shared<opset10::Concat>(OutputVector{conv1, conv2}, axis);
        model = std::make_shared<Model>(concat, ParameterVector{data});
        manager.register_pass<ov::pass::ConvolutionToGroupConvolutionFusion>();
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}
