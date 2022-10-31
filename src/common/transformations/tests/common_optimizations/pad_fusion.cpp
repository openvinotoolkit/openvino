// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>
#include <queue>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <transformations/common_optimizations/pad_fusion.hpp>
#include <transformations/common_optimizations/nop_elimination.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <ngraph/pass/manager.hpp>
#include <common_test_utils/ngraph_test_utils.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"


using namespace testing;
using namespace ngraph;

TEST_F(TransformationTestsF, PadElimination) {
    Shape data_shape{1, 3, 14, 14};
    {
        auto data = std::make_shared<opset5::Parameter>(element::i32, data_shape);
        auto pads_begin = opset5::Constant::create(element::i32, Shape{4}, {0, 0, 0, 0});
        auto pads_end = opset5::Constant::create(element::i32, Shape{4}, {0, 0, 0, 0});
        auto pad = std::make_shared<opset5::Pad>(data, pads_begin, pads_end, op::PadMode::CONSTANT);
        auto filters = std::make_shared<opset5::Parameter>(element::i32, Shape{1, 3, 4, 4});
        auto conv = std::make_shared<opset5::Convolution>(pad, filters, Strides{1, 1},
                                                          CoordinateDiff{0, 0}, CoordinateDiff{1, 1}, Shape{1, 1});
        function = std::make_shared<Function>(NodeVector{conv}, ParameterVector{data, filters});
        manager.register_pass<pass::EliminatePad>();
    }
    {
        auto data = std::make_shared<opset5::Parameter>(element::i32, data_shape);
        auto filters = std::make_shared<opset5::Parameter>(element::i32, Shape{1, 3, 4, 4});
        auto conv = std::make_shared<opset5::Convolution>(data, filters, Strides{1, 1},
                                                          CoordinateDiff{0, 0}, CoordinateDiff{1, 1}, Shape{1, 1},
                                                          op::PadType::EXPLICIT);
        function_ref = std::make_shared<Function>(NodeVector{conv}, ParameterVector{data, filters});
    }
}

TEST_F(TransformationTestsF, PadFusionAvgPoolExcludePad) {
    Shape data_shape{1, 3, 14, 14};
    {
        auto data = std::make_shared<opset5::Parameter>(element::i32, data_shape);
        auto pads_begin = opset5::Constant::create(element::i32, Shape{4}, {0, 0, 1, 1});
        auto pads_end = opset5::Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        auto pad = std::make_shared<opset5::Pad>(data, pads_begin, pads_end, op::PadMode::CONSTANT);
        auto avg_pool = std::make_shared<opset5::AvgPool>(pad, Strides{1, 1},
                                                          Shape{0, 0}, Shape{0, 0},
                                                          Shape{4, 4}, true, op::RoundingType::FLOOR);
        function = std::make_shared<Function>(NodeVector{avg_pool}, ParameterVector{data});
        manager.register_pass<pass::PadFusion>();
    }
    {
        auto data = std::make_shared<opset5::Parameter>(element::i32, data_shape);
        auto avg_pool = std::make_shared<opset5::AvgPool>(data, Strides{1, 1},
                                                          Shape{1, 1}, Shape{2, 2}, Shape{4, 4},
                                                          false, op::RoundingType::FLOOR, op::PadType::EXPLICIT);
        function_ref = std::make_shared<Function>(NodeVector{avg_pool}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, PadFusionAvgPoolDontExcludePad) {
    Shape data_shape{1, 3, 14, 14};
    {
        auto data = std::make_shared<opset5::Parameter>(element::i32, data_shape);
        auto pads_begin = opset5::Constant::create(element::i32, Shape{4}, {0, 0, 1, 1});
        auto pads_end = opset5::Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        auto pad = std::make_shared<opset5::Pad>(data, pads_begin, pads_end, op::PadMode::CONSTANT);
        auto avg_pool = std::make_shared<opset5::AvgPool>(pad, Strides{1, 1},
                                                          Shape{0, 0}, Shape{1, 1},
                                                          Shape{4, 4}, false, op::RoundingType::FLOOR);
        function = std::make_shared<Function>(NodeVector{avg_pool}, ParameterVector{data});
        manager.register_pass<pass::PadFusion>();
    }
    {
        auto data = std::make_shared<opset5::Parameter>(element::i32, data_shape);
        auto avg_pool = std::make_shared<opset5::AvgPool>(data, Strides{1, 1},
                                                          Shape{1, 1}, Shape{3, 3}, Shape{4, 4},
                                                          false, op::RoundingType::FLOOR, op::PadType::EXPLICIT);
        function_ref = std::make_shared<Function>(NodeVector{avg_pool}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, PadFusionConvolution) {
    Shape data_shape{1, 3, 14, 14};
    {
        auto data = std::make_shared<opset5::Parameter>(element::i32, data_shape);
        auto pads_begin = opset5::Constant::create(element::i32, Shape{4}, {0, 0, 1, 1});
        auto pads_end = opset5::Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        auto pad = std::make_shared<opset5::Pad>(data, pads_begin, pads_end, op::PadMode::CONSTANT);
        auto filters = std::make_shared<opset5::Parameter>(element::i32, Shape{1, 3, 4, 4});
        auto conv = std::make_shared<opset5::Convolution>(pad, filters, Strides{1, 1},
                                                          CoordinateDiff{0, 0}, CoordinateDiff{1, 1}, Shape{1, 1});
        function = std::make_shared<Function>(NodeVector{conv}, ParameterVector{data, filters});
        manager.register_pass<pass::PadFusion>();
    }
    {
        auto data = std::make_shared<opset5::Parameter>(element::i32, data_shape);
        auto filters = std::make_shared<opset5::Parameter>(element::i32, Shape{1, 3, 4, 4});
        auto conv = std::make_shared<opset5::Convolution>(data, filters, Strides{1, 1},
                                                          CoordinateDiff{1, 1}, CoordinateDiff{3, 3}, Shape{1, 1},
                                                          op::PadType::EXPLICIT);
        function_ref = std::make_shared<Function>(NodeVector{conv}, ParameterVector{data, filters});
    }
}

TEST_F(TransformationTestsF, PadFusionConvolutionBackpropData) {
    Shape data_shape{1, 3, 14, 14};
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
        auto pads_begin = opset5::Constant::create(element::i32, Shape{4}, {0, 0, 1, 1});
        auto pads_end = opset5::Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        auto pad = std::make_shared<opset5::Pad>(data, pads_begin, pads_end, op::PadMode::CONSTANT);

        auto filters = std::make_shared<opset5::Parameter>(element::f32, Shape{3, 2, 5, 5});
        auto conv = std::make_shared<opset5::ConvolutionBackpropData>(pad, filters, Strides{1, 1},
                                                                      CoordinateDiff{4, 4}, CoordinateDiff{3, 3}, Shape{1, 1});


        function = std::make_shared<Function>(NodeVector{conv}, ParameterVector{data, filters});
        manager.register_pass<pass::PadFusion>();
    }
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
        auto filters = std::make_shared<opset5::Parameter>(element::f32, Shape{3, 2, 5, 5});
        auto conv = std::make_shared<opset5::ConvolutionBackpropData>(data, filters, Strides{1, 1},
                                                                      CoordinateDiff{3, 3}, CoordinateDiff{1, 1}, Shape{1, 1});


        function_ref = std::make_shared<Function>(NodeVector{conv}, ParameterVector{data, filters});
    }
}

TEST_F(TransformationTestsF, PadFusionGroupConvolution) {
    Shape data_shape{1, 4, 14, 14};
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
        auto pads_begin = opset5::Constant::create(element::i32, Shape{4}, {0, 0, 1, 1});
        auto pads_end = opset5::Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        auto pad = std::make_shared<opset5::Pad>(data, pads_begin, pads_end, op::PadMode::CONSTANT);
        auto filters = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 4, 4, 4});
        auto conv = std::make_shared<opset5::GroupConvolution>(pad, filters, Strides{1, 1},
                                                               CoordinateDiff{0, 0}, CoordinateDiff{1, 1}, Shape{1, 1});

        function = std::make_shared<Function>(NodeVector{conv}, ParameterVector{data, filters});
        manager.register_pass<pass::PadFusion>();
    }
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
        auto filters = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 4, 4, 4});
        auto conv = std::make_shared<opset5::GroupConvolution>(data, filters, Strides{1, 1},
                                                               CoordinateDiff{1, 1}, CoordinateDiff{3, 3}, Shape{1, 1},
                                                               op::PadType::EXPLICIT);
        function_ref = std::make_shared<Function>(NodeVector{conv}, ParameterVector{data, filters});
    }
}

TEST_F(TransformationTestsF, PadFusionGroupConvolutionBackpropData) {
    Shape data_shape{1, 4, 14, 14};
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
        auto pads_begin = opset5::Constant::create(element::i32, Shape{4}, {0, 0, 1, 1});
        auto pads_end = opset5::Constant::create(element::i32, Shape{4}, {0, 0, 3, 1});
        auto pad = std::make_shared<opset5::Pad>(data, pads_begin, pads_end, op::PadMode::CONSTANT);
        auto filters = std::make_shared<opset5::Parameter>(element::f32, Shape{2, 2, 1, 5, 5});
        auto conv = std::make_shared<opset5::GroupConvolutionBackpropData>(pad, filters, Strides{1, 1},
                                                               CoordinateDiff{3, 2}, CoordinateDiff{4, 3}, Shape{1, 1});
        function = std::make_shared<Function>(NodeVector{conv}, ParameterVector{data, filters});
        manager.register_pass<pass::PadFusion>();
    }
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
        auto filters = std::make_shared<opset5::Parameter>(element::f32, Shape{2, 2, 1, 5, 5});
        auto conv = std::make_shared<opset5::GroupConvolutionBackpropData>(data, filters, Strides{1, 1},
                                                                           CoordinateDiff{2, 1}, CoordinateDiff{1, 2}, Shape{1, 1});
        function_ref = std::make_shared<Function>(NodeVector{conv}, ParameterVector{data, filters});
    }
}

TEST_F(TransformationTestsF, PadFusionAvgPoolNonConstPadValue) {
    Shape data_shape{1, 3, 14, 14};
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
        auto pads_begin = opset5::Constant::create(element::i32, Shape{4}, {0, 0, 1, 1});
        auto pads_end = opset5::Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        std::shared_ptr<Node> pad_value = opset5::Constant::create(element::f16, Shape{}, {0});
        pad_value = std::make_shared<opset5::Convert>(pad_value, element::f32);
        auto pad = std::make_shared<opset5::Pad>(data, pads_begin, pads_end, pad_value, op::PadMode::CONSTANT);
        auto avg_pool = std::make_shared<opset5::AvgPool>(pad, Strides{1, 1},
                                                          Shape{0, 0}, Shape{0, 0},
                                                          Shape{4, 4}, true, op::RoundingType::FLOOR);
        function = std::make_shared<Function>(NodeVector{avg_pool}, ParameterVector{data});
        manager.register_pass<pass::PadFusion>();
    }
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
        auto avg_pool = std::make_shared<opset5::AvgPool>(data, Strides{1, 1},
                                                          Shape{1, 1}, Shape{2, 2}, Shape{4, 4},
                                                          false, op::RoundingType::FLOOR, op::PadType::EXPLICIT);
        function_ref = std::make_shared<Function>(NodeVector{avg_pool}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, PadFusionConvolutionNonConstPadValue) {
    Shape data_shape{1, 3, 14, 14};
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
        auto pads_begin = opset5::Constant::create(element::i32, Shape{4}, {0, 0, 1, 1});
        auto pads_end = opset5::Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        std::shared_ptr<Node> pad_value = opset5::Constant::create(element::f16, Shape{}, {0});
        pad_value = std::make_shared<opset5::Convert>(pad_value, element::f32);
        auto pad = std::make_shared<opset5::Pad>(data, pads_begin, pads_end, pad_value, op::PadMode::CONSTANT);
        auto filters = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 3, 4, 4});
        auto conv = std::make_shared<opset5::Convolution>(pad, filters, Strides{1, 1},
                                                          CoordinateDiff{0, 0}, CoordinateDiff{1, 1}, Shape{1, 1});
        function = std::make_shared<Function>(NodeVector{conv}, ParameterVector{data, filters});
        manager.register_pass<pass::PadFusion>();
    }
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
        auto filters = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 3, 4, 4});
        auto conv = std::make_shared<opset5::Convolution>(data, filters, Strides{1, 1},
                                                          CoordinateDiff{1, 1}, CoordinateDiff{3, 3}, Shape{1, 1},
                                                          op::PadType::EXPLICIT);
        function_ref = std::make_shared<Function>(NodeVector{conv}, ParameterVector{data, filters});
    }
}

TEST_F(TransformationTestsF, PadFusionConvolutionBackpropDataNonConstPadValue) {
    Shape data_shape{1, 3, 14, 14};
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
        auto pads_begin = opset5::Constant::create(element::i32, Shape{4}, {0, 0, 1, 1});
        auto pads_end = opset5::Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        std::shared_ptr<Node> pad_value = opset5::Constant::create(element::f16, Shape{}, {0});
        pad_value = std::make_shared<opset5::Convert>(pad_value, element::f32);
        auto pad = std::make_shared<opset5::Pad>(data, pads_begin, pads_end, pad_value, op::PadMode::CONSTANT);

        auto filters = std::make_shared<opset5::Parameter>(element::f32, Shape{3, 2, 5, 5});
        auto conv = std::make_shared<opset5::ConvolutionBackpropData>(pad, filters, Strides{1, 1},
                                                                      CoordinateDiff{4, 4}, CoordinateDiff{3, 3}, Shape{1, 1});


        function = std::make_shared<Function>(NodeVector{conv}, ParameterVector{data, filters});
        manager.register_pass<pass::PadFusion>();
    }
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
        auto filters = std::make_shared<opset5::Parameter>(element::f32, Shape{3, 2, 5, 5});
        auto conv = std::make_shared<opset5::ConvolutionBackpropData>(data, filters, Strides{1, 1},
                                                                      CoordinateDiff{3, 3}, CoordinateDiff{1, 1}, Shape{1, 1});


        function_ref = std::make_shared<Function>(NodeVector{conv}, ParameterVector{data, filters});
    }
}

TEST_F(TransformationTestsF, PadFusionGroupConvolutionNonConstPadValue) {
    Shape data_shape{1, 4, 14, 14};
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
        auto pads_begin = opset5::Constant::create(element::i32, Shape{4}, {0, 0, 1, 1});
        auto pads_end = opset5::Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        std::shared_ptr<Node> pad_value = opset5::Constant::create(element::f16, Shape{}, {0});
        pad_value = std::make_shared<opset5::Convert>(pad_value, element::f32);
        auto pad = std::make_shared<opset5::Pad>(data, pads_begin, pads_end, pad_value, op::PadMode::CONSTANT);
        auto filters = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 4, 4, 4});
        auto conv = std::make_shared<opset5::GroupConvolution>(pad, filters, Strides{1, 1},
                                                               CoordinateDiff{0, 0}, CoordinateDiff{1, 1}, Shape{1, 1});

        function = std::make_shared<Function>(NodeVector{conv}, ParameterVector{data, filters});
        manager.register_pass<pass::PadFusion>();
    }
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
        auto filters = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 4, 4, 4});
        auto conv = std::make_shared<opset5::GroupConvolution>(data, filters, Strides{1, 1},
                                                               CoordinateDiff{1, 1}, CoordinateDiff{3, 3}, Shape{1, 1},
                                                               op::PadType::EXPLICIT);
        function_ref = std::make_shared<Function>(NodeVector{conv}, ParameterVector{data, filters});
    }
}

TEST_F(TransformationTestsF, PadFusionGroupConvolutionBackpropDataNonConstPadValue) {
    Shape data_shape{1, 4, 14, 14};
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
        auto pads_begin = opset5::Constant::create(element::i32, Shape{4}, {0, 0, 1, 1});
        auto pads_end = opset5::Constant::create(element::i32, Shape{4}, {0, 0, 3, 1});
        std::shared_ptr<Node> pad_value = opset5::Constant::create(element::f16, Shape{}, {0});
        pad_value = std::make_shared<opset5::Convert>(pad_value, element::f32);
        auto pad = std::make_shared<opset5::Pad>(data, pads_begin, pads_end, pad_value, op::PadMode::CONSTANT);
        auto filters = std::make_shared<opset5::Parameter>(element::f32, Shape{2, 2, 1, 5, 5});
        auto conv = std::make_shared<opset5::GroupConvolutionBackpropData>(pad, filters, Strides{1, 1},
                                                               CoordinateDiff{3, 2}, CoordinateDiff{4, 3}, Shape{1, 1});
        function = std::make_shared<Function>(NodeVector{conv}, ParameterVector{data, filters});
        manager.register_pass<pass::PadFusion>();
    }
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
        auto filters = std::make_shared<opset5::Parameter>(element::f32, Shape{2, 2, 1, 5, 5});
        auto conv = std::make_shared<opset5::GroupConvolutionBackpropData>(data, filters, Strides{1, 1},
                                                                           CoordinateDiff{2, 1}, CoordinateDiff{1, 2}, Shape{1, 1});
        function_ref = std::make_shared<Function>(NodeVector{conv}, ParameterVector{data, filters});
    }
}

TEST_F(TransformationTestsF, NegativePadFusionNonConstantPadMode) {
    Shape data_shape{1, 3, 14, 14};
    {
        auto data = std::make_shared<opset5::Parameter>(element::i32, data_shape);
        auto pads_begin = opset5::Constant::create(element::i32, Shape{4}, {0, 0, 1, 1});
        auto pads_end = opset5::Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        auto pad = std::make_shared<opset5::Pad>(data, pads_begin, pads_end, op::PadMode::REFLECT);
        auto filters = std::make_shared<opset5::Parameter>(element::i32, Shape{1, 3, 4, 4});
        auto conv = std::make_shared<opset5::Convolution>(pad, filters, Strides{1, 1},
                                                          CoordinateDiff{0, 0}, CoordinateDiff{1, 1}, Shape{1, 1});
        function = std::make_shared<Function>(NodeVector{conv}, ParameterVector{data, filters});
        manager.register_pass<pass::PadFusion>();
    }
    {
        auto data = std::make_shared<opset5::Parameter>(element::i32, data_shape);
        auto pads_begin = opset5::Constant::create(element::i32, Shape{4}, {0, 0, 1, 1});
        auto pads_end = opset5::Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        auto pad = std::make_shared<opset5::Pad>(data, pads_begin, pads_end, op::PadMode::REFLECT);
        auto filters = std::make_shared<opset5::Parameter>(element::i32, Shape{1, 3, 4, 4});
        auto conv = std::make_shared<opset5::Convolution>(pad, filters, Strides{1, 1},
                                                          CoordinateDiff{0, 0}, CoordinateDiff{1, 1}, Shape{1, 1});
        function_ref = std::make_shared<Function>(NodeVector{conv}, ParameterVector{data, filters});
    }
}

TEST_F(TransformationTestsF, NegativePadFusionNonZeroPadValue) {
    Shape data_shape{1, 3, 14, 14};
    {
        auto data = std::make_shared<opset5::Parameter>(element::i32, data_shape);
        auto pads_begin = opset5::Constant::create(element::i32, Shape{4}, {0, 0, 1, 1});
        auto pads_end = opset5::Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        auto pad_value = opset5::Constant::create(element::i32, Shape{}, {2});
        auto pad = std::make_shared<opset5::Pad>(data, pads_begin, pads_end, pad_value, op::PadMode::CONSTANT);
        auto filters = std::make_shared<opset5::Parameter>(element::i32, Shape{1, 3, 4, 4});
        auto conv = std::make_shared<opset5::Convolution>(pad, filters, Strides{1, 1},
                                                          CoordinateDiff{0, 0}, CoordinateDiff{1, 1}, Shape{1, 1});
        function = std::make_shared<Function>(NodeVector{conv}, ParameterVector{data, filters});
        manager.register_pass<pass::PadFusion>();
    }
    {
        auto data = std::make_shared<opset5::Parameter>(element::i32, data_shape);
        auto pads_begin = opset5::Constant::create(element::i32, Shape{4}, {0, 0, 1, 1});
        auto pads_end = opset5::Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        auto pad_value = opset5::Constant::create(element::i32, Shape{}, {2});
        auto pad = std::make_shared<opset5::Pad>(data, pads_begin, pads_end, pad_value, op::PadMode::CONSTANT);
        auto filters = std::make_shared<opset5::Parameter>(element::i32, Shape{1, 3, 4, 4});
        auto conv = std::make_shared<opset5::Convolution>(pad, filters, Strides{1, 1},
                                                          CoordinateDiff{0, 0}, CoordinateDiff{1, 1}, Shape{1, 1});
        function_ref = std::make_shared<Function>(NodeVector{conv}, ParameterVector{data, filters});
    }
}

TEST_F(TransformationTestsF, NegativePadFusionPadForBatchSize) {
    Shape data_shape{1, 3, 14, 14};
    {
        auto data = std::make_shared<opset5::Parameter>(element::i32, data_shape);
        auto pads_begin = opset5::Constant::create(element::i32, Shape{4}, {1, 0, 1, 1});
        auto pads_end = opset5::Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        auto pad_value = opset5::Constant::create(element::i32, Shape{}, {0});
        auto pad = std::make_shared<opset5::Pad>(data, pads_begin, pads_end, pad_value, op::PadMode::CONSTANT);
        auto filters = std::make_shared<opset5::Parameter>(element::i32, Shape{1, 3, 4, 4});
        auto conv = std::make_shared<opset5::Convolution>(pad, filters, Strides{1, 1},
                                                          CoordinateDiff{0, 0}, CoordinateDiff{1, 1}, Shape{1, 1});
        function = std::make_shared<Function>(NodeVector{conv}, ParameterVector{data, filters});
        manager.register_pass<pass::PadFusion>();
    }
    {
        auto data = std::make_shared<opset5::Parameter>(element::i32, data_shape);
        auto pads_begin = opset5::Constant::create(element::i32, Shape{4}, {1, 0, 1, 1});
        auto pads_end = opset5::Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        auto pad_value = opset5::Constant::create(element::i32, Shape{}, {0});
        auto pad = std::make_shared<opset5::Pad>(data, pads_begin, pads_end, pad_value, op::PadMode::CONSTANT);
        auto filters = std::make_shared<opset5::Parameter>(element::i32, Shape{1, 3, 4, 4});
        auto conv = std::make_shared<opset5::Convolution>(pad, filters, Strides{1, 1},
                                                          CoordinateDiff{0, 0}, CoordinateDiff{1, 1}, Shape{1, 1});
        function_ref = std::make_shared<Function>(NodeVector{conv}, ParameterVector{data, filters});
    }
}

TEST_F(TransformationTestsF, NegativePadFusionAvgPoolExcludePadNonZeroPads) {
    Shape data_shape{1, 3, 14, 14};
    {
        auto data = std::make_shared<opset5::Parameter>(element::i32, data_shape);
        auto pads_begin = opset5::Constant::create(element::i32, Shape{4}, {0, 0, 1, 1});
        auto pads_end = opset5::Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        auto pad = std::make_shared<opset5::Pad>(data, pads_begin, pads_end, op::PadMode::CONSTANT);
        auto avg_pool = std::make_shared<opset5::AvgPool>(pad, Strides{1, 1},
                                                          Shape{0, 0}, Shape{1, 1},
                                                          Shape{4, 4}, true, op::RoundingType::FLOOR);
        function = std::make_shared<Function>(NodeVector{avg_pool}, ParameterVector{data});
        manager.register_pass<pass::PadFusion>();
    }
    {
        auto data = std::make_shared<opset5::Parameter>(element::i32, data_shape);
        auto pads_begin = opset5::Constant::create(element::i32, Shape{4}, {0, 0, 1, 1});
        auto pads_end = opset5::Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        auto pad = std::make_shared<opset5::Pad>(data, pads_begin, pads_end, op::PadMode::CONSTANT);
        auto avg_pool = std::make_shared<opset5::AvgPool>(pad, Strides{1, 1},
                                                          Shape{0, 0}, Shape{1, 1},
                                                          Shape{4, 4}, true, op::RoundingType::FLOOR);
        function_ref = std::make_shared<Function>(NodeVector{avg_pool}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, NegativePadFusionConvolutionBackpropDataTooSmallPad) {
    Shape data_shape{1, 3, 14, 14};
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);

        auto pads_begin = opset5::Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        auto pads_end = opset5::Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        auto pad = std::make_shared<opset5::Pad>(data, pads_begin, pads_end, op::PadMode::CONSTANT);

        auto filters = std::make_shared<opset5::Parameter>(element::f32, Shape{3, 2, 5, 5});
        auto conv = std::make_shared<opset5::ConvolutionBackpropData>(pad, filters, Strides{1, 1},
                                                                      CoordinateDiff{1, 1}, CoordinateDiff{1, 1}, Shape{1, 1});


        function = std::make_shared<Function>(NodeVector{conv}, ParameterVector{data, filters});
        manager.register_pass<pass::PadFusion>();
    }
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);

        auto pads_begin = opset5::Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        auto pads_end = opset5::Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        auto pad = std::make_shared<opset5::Pad>(data, pads_begin, pads_end, op::PadMode::CONSTANT);

        auto filters = std::make_shared<opset5::Parameter>(element::f32, Shape{3, 2, 5, 5});
        auto conv = std::make_shared<opset5::ConvolutionBackpropData>(pad, filters, Strides{1, 1},
                                                                      CoordinateDiff{1, 1}, CoordinateDiff{1, 1}, Shape{1, 1});

        function_ref = std::make_shared<Function>(NodeVector{conv}, ParameterVector{data, filters});
    }
}
