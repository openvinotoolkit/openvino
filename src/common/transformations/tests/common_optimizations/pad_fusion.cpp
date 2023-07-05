// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <common_test_utils/ngraph_test_utils.hpp>
#include <memory>
#include <openvino/op/pad.hpp>
#include <queue>
#include <string>
#include <transformations/common_optimizations/nop_elimination.hpp>
#include <transformations/common_optimizations/pad_fusion.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset12.hpp"
#include "openvino/pass/manager.hpp"
#include "pad_test_utils.hpp"

using namespace testing;
using namespace ov;
using namespace ov::opset12;

using NodePtr = std::shared_ptr<ov::Node>;
using PadFactoryPtr = std::shared_ptr<IPadFactory>;
using TestModelFactoryPtr = std::shared_ptr<ITestModelFactory>;
using TestParams = std::tuple<PadFactoryPtr, TestModelFactoryPtr>;

PAD_TEST_BODY(PadElimination) {
    Shape data_shape{1, 3, 14, 14};
    {
        auto data = std::make_shared<Parameter>(element::i32, data_shape);
        auto pads_begin = Constant::create(element::i32, Shape{4}, {0, 0, 0, 0});
        auto pads_end = Constant::create(element::i32, Shape{4}, {0, 0, 0, 0});
        auto pad = pad_factory->create(data, pads_begin, pads_end, op::PadMode::CONSTANT);
        auto filters = std::make_shared<Parameter>(element::i32, Shape{1, 3, 4, 4});
        auto conv = std::make_shared<Convolution>(pad,
                                                  filters,
                                                  Strides{1, 1},
                                                  CoordinateDiff{0, 0},
                                                  CoordinateDiff{1, 1},
                                                  Shape{1, 1});
        function = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data, filters});
        manager.register_pass<ov::pass::EliminatePad>();
    }
    {
        auto data = std::make_shared<Parameter>(element::i32, data_shape);
        auto filters = std::make_shared<Parameter>(element::i32, Shape{1, 3, 4, 4});
        auto conv = std::make_shared<Convolution>(data,
                                                  filters,
                                                  Strides{1, 1},
                                                  CoordinateDiff{0, 0},
                                                  CoordinateDiff{1, 1},
                                                  Shape{1, 1},
                                                  op::PadType::EXPLICIT);
        function_ref = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data, filters});
    }
}

PAD_TEST_BODY(NegativePadElimination) {
    Shape data_shape{1, 3, 14, 14};
    {
        auto data = std::make_shared<Parameter>(element::i32, data_shape);
        auto pads_begin = Constant::create(element::i32, Shape{4}, {0, 0, -1, -1});
        auto pads_end = Constant::create(element::i32, Shape{4}, {0, 0, -1, -1});
        auto pad = pad_factory->create(data, pads_begin, pads_end, op::PadMode::CONSTANT);
        auto filters = std::make_shared<Parameter>(element::i32, Shape{1, 3, 4, 4});
        auto conv = std::make_shared<Convolution>(pad,
                                                  filters,
                                                  Strides{1, 1},
                                                  CoordinateDiff{0, 0},
                                                  CoordinateDiff{1, 1},
                                                  Shape{1, 1});
        function = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data, filters});
        manager.register_pass<ov::pass::EliminatePad>();
    }
    // Reference function is equal to function
}

PAD_TEST_BODY(PadFusionAvgPoolExcludePad) {
    Shape data_shape{1, 3, 14, 14};
    {
        auto data = std::make_shared<Parameter>(element::i32, data_shape);
        auto pads_begin = Constant::create(element::i32, Shape{4}, {0, 0, 1, 1});
        auto pads_end = Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        auto pad = pad_factory->create(data, pads_begin, pads_end, op::PadMode::CONSTANT);
        auto avg_pool = std::make_shared<AvgPool>(pad,
                                                  Strides{1, 1},
                                                  Shape{0, 0},
                                                  Shape{0, 0},
                                                  Shape{4, 4},
                                                  true,
                                                  op::RoundingType::FLOOR);
        function = std::make_shared<Model>(NodeVector{avg_pool}, ParameterVector{data});
        manager.register_pass<ov::pass::PadFusion>();
    }
    {
        auto data = std::make_shared<Parameter>(element::i32, data_shape);
        auto avg_pool = std::make_shared<AvgPool>(data,
                                                  Strides{1, 1},
                                                  Shape{1, 1},
                                                  Shape{2, 2},
                                                  Shape{4, 4},
                                                  false,
                                                  op::RoundingType::FLOOR,
                                                  op::PadType::EXPLICIT);
        function_ref = std::make_shared<Model>(NodeVector{avg_pool}, ParameterVector{data});
    }
}

PAD_TEST_BODY(NegativePadFusionAvgPoolExcludePad) {
    Shape data_shape{1, 3, 14, 14};
    {
        auto data = std::make_shared<Parameter>(element::i32, data_shape);
        auto pads_begin = Constant::create(element::i32, Shape{4}, {0, 0, -1, -1});
        auto pads_end = Constant::create(element::i32, Shape{4}, {0, 0, -2, -2});
        auto pad = pad_factory->create(data, pads_begin, pads_end, op::PadMode::CONSTANT);
        auto avg_pool = std::make_shared<AvgPool>(pad,
                                                  Strides{1, 1},
                                                  Shape{0, 0},
                                                  Shape{0, 0},
                                                  Shape{4, 4},
                                                  true,
                                                  op::RoundingType::FLOOR);
        function = std::make_shared<Model>(NodeVector{avg_pool}, ParameterVector{data});
        manager.register_pass<ov::pass::PadFusion>();
    }
    // Reference function is equal to function
}

PAD_TEST_BODY(PadFusionAvgPoolDontExcludePad) {
    Shape data_shape{1, 3, 14, 14};
    {
        auto data = std::make_shared<Parameter>(element::i32, data_shape);
        auto pads_begin = Constant::create(element::i32, Shape{4}, {0, 0, 1, 1});
        auto pads_end = Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        auto pad = pad_factory->create(data, pads_begin, pads_end, op::PadMode::CONSTANT);
        auto avg_pool = std::make_shared<AvgPool>(pad,
                                                  Strides{1, 1},
                                                  Shape{0, 0},
                                                  Shape{1, 1},
                                                  Shape{4, 4},
                                                  false,
                                                  op::RoundingType::FLOOR);
        function = std::make_shared<Model>(NodeVector{avg_pool}, ParameterVector{data});
        manager.register_pass<ov::pass::PadFusion>();
    }
    {
        auto data = std::make_shared<Parameter>(element::i32, data_shape);
        auto avg_pool = std::make_shared<AvgPool>(data,
                                                  Strides{1, 1},
                                                  Shape{1, 1},
                                                  Shape{3, 3},
                                                  Shape{4, 4},
                                                  false,
                                                  op::RoundingType::FLOOR,
                                                  op::PadType::EXPLICIT);
        function_ref = std::make_shared<Model>(NodeVector{avg_pool}, ParameterVector{data});
    }
}

PAD_TEST_BODY(NegativePadFusionAvgPoolDontExcludePad) {
    Shape data_shape{1, 3, 14, 14};
    {
        auto data = std::make_shared<Parameter>(element::i32, data_shape);
        auto pads_begin = Constant::create(element::i32, Shape{4}, {0, 0, -1, -1});
        auto pads_end = Constant::create(element::i32, Shape{4}, {0, 0, -2, -2});
        auto pad = pad_factory->create(data, pads_begin, pads_end, op::PadMode::CONSTANT);
        auto avg_pool = std::make_shared<AvgPool>(pad,
                                                  Strides{1, 1},
                                                  Shape{0, 0},
                                                  Shape{1, 1},
                                                  Shape{4, 4},
                                                  false,
                                                  op::RoundingType::FLOOR);
        function = std::make_shared<Model>(NodeVector{avg_pool}, ParameterVector{data});
        manager.register_pass<ov::pass::PadFusion>();
    }
    // Reference function is equal to function
}

PAD_TEST_BODY(PadFusionConvolution) {
    Shape data_shape{1, 3, 14, 14};
    {
        auto data = std::make_shared<Parameter>(element::i32, data_shape);
        auto pads_begin = Constant::create(element::i32, Shape{4}, {0, 0, 1, 1});
        auto pads_end = Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        auto pad = pad_factory->create(data, pads_begin, pads_end, op::PadMode::CONSTANT);
        auto filters = std::make_shared<Parameter>(element::i32, Shape{1, 3, 4, 4});
        auto conv = std::make_shared<Convolution>(pad,
                                                  filters,
                                                  Strides{1, 1},
                                                  CoordinateDiff{0, 0},
                                                  CoordinateDiff{1, 1},
                                                  Shape{1, 1});
        function = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data, filters});
        manager.register_pass<ov::pass::PadFusion>();
    }
    {
        auto data = std::make_shared<Parameter>(element::i32, data_shape);
        auto filters = std::make_shared<Parameter>(element::i32, Shape{1, 3, 4, 4});
        auto conv = std::make_shared<Convolution>(data,
                                                  filters,
                                                  Strides{1, 1},
                                                  CoordinateDiff{1, 1},
                                                  CoordinateDiff{3, 3},
                                                  Shape{1, 1},
                                                  op::PadType::EXPLICIT);
        function_ref = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data, filters});
    }
}

PAD_TEST_BODY(NegativePadFusionConvolution) {
    Shape data_shape{1, 3, 14, 14};
    {
        auto data = std::make_shared<Parameter>(element::i32, data_shape);
        auto pads_begin = Constant::create(element::i32, Shape{4}, {0, 0, -1, -1});
        auto pads_end = Constant::create(element::i32, Shape{4}, {0, 0, -2, -2});
        auto pad = pad_factory->create(data, pads_begin, pads_end, op::PadMode::CONSTANT);
        auto filters = std::make_shared<Parameter>(element::i32, Shape{1, 3, 4, 4});
        auto conv = std::make_shared<Convolution>(pad,
                                                  filters,
                                                  Strides{1, 1},
                                                  CoordinateDiff{0, 0},
                                                  CoordinateDiff{1, 1},
                                                  Shape{1, 1});
        function = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data, filters});
        manager.register_pass<ov::pass::PadFusion>();
    }
    // Reference function is equal to function
}

PAD_TEST_BODY(PadFusionConvolutionBackpropData) {
    Shape data_shape{1, 3, 14, 14};
    {
        auto data = std::make_shared<Parameter>(element::f32, data_shape);
        auto pads_begin = Constant::create(element::i32, Shape{4}, {0, 0, 1, 1});
        auto pads_end = Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        auto pad = pad_factory->create(data, pads_begin, pads_end, op::PadMode::CONSTANT);

        auto filters = std::make_shared<Parameter>(element::f32, Shape{3, 2, 5, 5});
        auto conv = std::make_shared<ConvolutionBackpropData>(pad,
                                                              filters,
                                                              Strides{1, 1},
                                                              CoordinateDiff{4, 4},
                                                              CoordinateDiff{3, 3},
                                                              Shape{1, 1});

        function = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data, filters});
        manager.register_pass<ov::pass::PadFusion>();
    }
    {
        auto data = std::make_shared<Parameter>(element::f32, data_shape);
        auto filters = std::make_shared<Parameter>(element::f32, Shape{3, 2, 5, 5});
        auto conv = std::make_shared<ConvolutionBackpropData>(data,
                                                              filters,
                                                              Strides{1, 1},
                                                              CoordinateDiff{3, 3},
                                                              CoordinateDiff{1, 1},
                                                              Shape{1, 1});

        function_ref = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data, filters});
    }
}

PAD_TEST_BODY(PadFusionGroupConvolution) {
    Shape data_shape{1, 4, 14, 14};
    {
        auto data = std::make_shared<Parameter>(element::f32, data_shape);
        auto pads_begin = Constant::create(element::i32, Shape{4}, {0, 0, 1, 1});
        auto pads_end = Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        auto pad = pad_factory->create(data, pads_begin, pads_end, op::PadMode::CONSTANT);
        auto filters = std::make_shared<Parameter>(element::f32, Shape{1, 1, 4, 4, 4});
        auto conv = std::make_shared<GroupConvolution>(pad,
                                                       filters,
                                                       Strides{1, 1},
                                                       CoordinateDiff{0, 0},
                                                       CoordinateDiff{1, 1},
                                                       Shape{1, 1});

        function = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data, filters});
        manager.register_pass<ov::pass::PadFusion>();
    }
    {
        auto data = std::make_shared<Parameter>(element::f32, data_shape);
        auto filters = std::make_shared<Parameter>(element::f32, Shape{1, 1, 4, 4, 4});
        auto conv = std::make_shared<GroupConvolution>(data,
                                                       filters,
                                                       Strides{1, 1},
                                                       CoordinateDiff{1, 1},
                                                       CoordinateDiff{3, 3},
                                                       Shape{1, 1},
                                                       op::PadType::EXPLICIT);
        function_ref = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data, filters});
    }
}

PAD_TEST_BODY(NegativePadFusionGroupConvolution) {
    Shape data_shape{1, 4, 14, 14};
    {
        auto data = std::make_shared<Parameter>(element::f32, data_shape);
        auto pads_begin = Constant::create(element::i32, Shape{4}, {0, 0, -1, -1});
        auto pads_end = Constant::create(element::i32, Shape{4}, {0, 0, -2, -2});
        auto pad = pad_factory->create(data, pads_begin, pads_end, op::PadMode::CONSTANT);
        auto filters = std::make_shared<Parameter>(element::f32, Shape{1, 1, 4, 4, 4});
        auto conv = std::make_shared<GroupConvolution>(pad,
                                                       filters,
                                                       Strides{1, 1},
                                                       CoordinateDiff{0, 0},
                                                       CoordinateDiff{1, 1},
                                                       Shape{1, 1});

        function = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data, filters});
        manager.register_pass<ov::pass::PadFusion>();
    }
    // Reference function is equal to function
}

PAD_TEST_BODY(PadFusionGroupConvolutionBackpropData) {
    Shape data_shape{1, 4, 14, 14};
    {
        auto data = std::make_shared<Parameter>(element::f32, data_shape);
        auto pads_begin = Constant::create(element::i32, Shape{4}, {0, 0, 1, 1});
        auto pads_end = Constant::create(element::i32, Shape{4}, {0, 0, 3, 1});
        auto pad = pad_factory->create(data, pads_begin, pads_end, op::PadMode::CONSTANT);
        auto filters = std::make_shared<Parameter>(element::f32, Shape{2, 2, 1, 5, 5});
        auto conv = std::make_shared<GroupConvolutionBackpropData>(pad,
                                                                   filters,
                                                                   Strides{1, 1},
                                                                   CoordinateDiff{3, 2},
                                                                   CoordinateDiff{4, 3},
                                                                   Shape{1, 1});
        function = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data, filters});
        manager.register_pass<ov::pass::PadFusion>();
    }
    {
        auto data = std::make_shared<Parameter>(element::f32, data_shape);
        auto filters = std::make_shared<Parameter>(element::f32, Shape{2, 2, 1, 5, 5});
        auto conv = std::make_shared<GroupConvolutionBackpropData>(data,
                                                                   filters,
                                                                   Strides{1, 1},
                                                                   CoordinateDiff{2, 1},
                                                                   CoordinateDiff{1, 2},
                                                                   Shape{1, 1});
        function_ref = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data, filters});
    }
}

PAD_TEST_BODY(PadFusionAvgPoolNonConstPadValue) {
    Shape data_shape{1, 3, 14, 14};
    {
        auto data = std::make_shared<Parameter>(element::f32, data_shape);
        auto pads_begin = Constant::create(element::i32, Shape{4}, {0, 0, 1, 1});
        auto pads_end = Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        std::shared_ptr<Node> pad_value = Constant::create(element::f16, Shape{}, {0});
        pad_value = std::make_shared<Convert>(pad_value, element::f32);
        auto pad = pad_factory->create(data, pads_begin, pads_end, pad_value, op::PadMode::CONSTANT);
        auto avg_pool = std::make_shared<AvgPool>(pad,
                                                  Strides{1, 1},
                                                  Shape{0, 0},
                                                  Shape{0, 0},
                                                  Shape{4, 4},
                                                  true,
                                                  op::RoundingType::FLOOR);
        function = std::make_shared<Model>(NodeVector{avg_pool}, ParameterVector{data});
        manager.register_pass<ov::pass::PadFusion>();
    }
    {
        auto data = std::make_shared<Parameter>(element::f32, data_shape);
        auto avg_pool = std::make_shared<AvgPool>(data,
                                                  Strides{1, 1},
                                                  Shape{1, 1},
                                                  Shape{2, 2},
                                                  Shape{4, 4},
                                                  false,
                                                  op::RoundingType::FLOOR,
                                                  op::PadType::EXPLICIT);
        function_ref = std::make_shared<Model>(NodeVector{avg_pool}, ParameterVector{data});
    }
}

PAD_TEST_BODY(PadFusionConvolutionNonConstPadValue) {
    Shape data_shape{1, 3, 14, 14};
    {
        auto data = std::make_shared<Parameter>(element::f32, data_shape);
        auto pads_begin = Constant::create(element::i32, Shape{4}, {0, 0, 1, 1});
        auto pads_end = Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        std::shared_ptr<Node> pad_value = Constant::create(element::f16, Shape{}, {0});
        pad_value = std::make_shared<Convert>(pad_value, element::f32);
        auto pad = pad_factory->create(data, pads_begin, pads_end, pad_value, op::PadMode::CONSTANT);
        auto filters = std::make_shared<Parameter>(element::f32, Shape{1, 3, 4, 4});
        auto conv = std::make_shared<Convolution>(pad,
                                                  filters,
                                                  Strides{1, 1},
                                                  CoordinateDiff{0, 0},
                                                  CoordinateDiff{1, 1},
                                                  Shape{1, 1});
        function = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data, filters});
        manager.register_pass<ov::pass::PadFusion>();
    }
    {
        auto data = std::make_shared<Parameter>(element::f32, data_shape);
        auto filters = std::make_shared<Parameter>(element::f32, Shape{1, 3, 4, 4});
        auto conv = std::make_shared<Convolution>(data,
                                                  filters,
                                                  Strides{1, 1},
                                                  CoordinateDiff{1, 1},
                                                  CoordinateDiff{3, 3},
                                                  Shape{1, 1},
                                                  op::PadType::EXPLICIT);
        function_ref = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data, filters});
    }
}

PAD_TEST_BODY(PadFusionConvolutionBackpropDataNonConstPadValue) {
    Shape data_shape{1, 3, 14, 14};
    {
        auto data = std::make_shared<Parameter>(element::f32, data_shape);
        auto pads_begin = Constant::create(element::i32, Shape{4}, {0, 0, 1, 1});
        auto pads_end = Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        std::shared_ptr<Node> pad_value = Constant::create(element::f16, Shape{}, {0});
        pad_value = std::make_shared<Convert>(pad_value, element::f32);
        auto pad = pad_factory->create(data, pads_begin, pads_end, pad_value, op::PadMode::CONSTANT);

        auto filters = std::make_shared<Parameter>(element::f32, Shape{3, 2, 5, 5});
        auto conv = std::make_shared<ConvolutionBackpropData>(pad,
                                                              filters,
                                                              Strides{1, 1},
                                                              CoordinateDiff{4, 4},
                                                              CoordinateDiff{3, 3},
                                                              Shape{1, 1});

        function = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data, filters});
        manager.register_pass<ov::pass::PadFusion>();
    }
    {
        auto data = std::make_shared<Parameter>(element::f32, data_shape);
        auto filters = std::make_shared<Parameter>(element::f32, Shape{3, 2, 5, 5});
        auto conv = std::make_shared<ConvolutionBackpropData>(data,
                                                              filters,
                                                              Strides{1, 1},
                                                              CoordinateDiff{3, 3},
                                                              CoordinateDiff{1, 1},
                                                              Shape{1, 1});

        function_ref = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data, filters});
    }
}

PAD_TEST_BODY(PadFusionGroupConvolutionNonConstPadValue) {
    Shape data_shape{1, 4, 14, 14};
    {
        auto data = std::make_shared<Parameter>(element::f32, data_shape);
        auto pads_begin = Constant::create(element::i32, Shape{4}, {0, 0, 1, 1});
        auto pads_end = Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        std::shared_ptr<Node> pad_value = Constant::create(element::f16, Shape{}, {0});
        pad_value = std::make_shared<Convert>(pad_value, element::f32);
        auto pad = pad_factory->create(data, pads_begin, pads_end, pad_value, op::PadMode::CONSTANT);
        auto filters = std::make_shared<Parameter>(element::f32, Shape{1, 1, 4, 4, 4});
        auto conv = std::make_shared<GroupConvolution>(pad,
                                                       filters,
                                                       Strides{1, 1},
                                                       CoordinateDiff{0, 0},
                                                       CoordinateDiff{1, 1},
                                                       Shape{1, 1});

        function = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data, filters});
        manager.register_pass<ov::pass::PadFusion>();
    }
    {
        auto data = std::make_shared<Parameter>(element::f32, data_shape);
        auto filters = std::make_shared<Parameter>(element::f32, Shape{1, 1, 4, 4, 4});
        auto conv = std::make_shared<GroupConvolution>(data,
                                                       filters,
                                                       Strides{1, 1},
                                                       CoordinateDiff{1, 1},
                                                       CoordinateDiff{3, 3},
                                                       Shape{1, 1},
                                                       op::PadType::EXPLICIT);
        function_ref = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data, filters});
    }
}

PAD_TEST_BODY(PadFusionGroupConvolutionBackpropDataNonConstPadValue) {
    Shape data_shape{1, 4, 14, 14};
    {
        auto data = std::make_shared<Parameter>(element::f32, data_shape);
        auto pads_begin = Constant::create(element::i32, Shape{4}, {0, 0, 1, 1});
        auto pads_end = Constant::create(element::i32, Shape{4}, {0, 0, 3, 1});
        std::shared_ptr<Node> pad_value = Constant::create(element::f16, Shape{}, {0});
        pad_value = std::make_shared<Convert>(pad_value, element::f32);
        auto pad = pad_factory->create(data, pads_begin, pads_end, pad_value, op::PadMode::CONSTANT);
        auto filters = std::make_shared<Parameter>(element::f32, Shape{2, 2, 1, 5, 5});
        auto conv = std::make_shared<GroupConvolutionBackpropData>(pad,
                                                                   filters,
                                                                   Strides{1, 1},
                                                                   CoordinateDiff{3, 2},
                                                                   CoordinateDiff{4, 3},
                                                                   Shape{1, 1});
        function = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data, filters});
        manager.register_pass<ov::pass::PadFusion>();
    }
    {
        auto data = std::make_shared<Parameter>(element::f32, data_shape);
        auto filters = std::make_shared<Parameter>(element::f32, Shape{2, 2, 1, 5, 5});
        auto conv = std::make_shared<GroupConvolutionBackpropData>(data,
                                                                   filters,
                                                                   Strides{1, 1},
                                                                   CoordinateDiff{2, 1},
                                                                   CoordinateDiff{1, 2},
                                                                   Shape{1, 1});
        function_ref = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data, filters});
    }
}

PAD_TEST_BODY(NegativePadFusionNonConstantPadMode) {
    Shape data_shape{1, 3, 14, 14};
    {
        auto data = std::make_shared<Parameter>(element::i32, data_shape);
        auto pads_begin = Constant::create(element::i32, Shape{4}, {0, 0, 1, 1});
        auto pads_end = Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        auto pad = pad_factory->create(data, pads_begin, pads_end, op::PadMode::REFLECT);
        auto filters = std::make_shared<Parameter>(element::i32, Shape{1, 3, 4, 4});
        auto conv = std::make_shared<Convolution>(pad,
                                                  filters,
                                                  Strides{1, 1},
                                                  CoordinateDiff{0, 0},
                                                  CoordinateDiff{1, 1},
                                                  Shape{1, 1});
        function = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data, filters});
        manager.register_pass<ov::pass::PadFusion>();
    }
    {
        auto data = std::make_shared<Parameter>(element::i32, data_shape);
        auto pads_begin = Constant::create(element::i32, Shape{4}, {0, 0, 1, 1});
        auto pads_end = Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        auto pad = pad_factory->create(data, pads_begin, pads_end, op::PadMode::REFLECT);
        auto filters = std::make_shared<Parameter>(element::i32, Shape{1, 3, 4, 4});
        auto conv = std::make_shared<Convolution>(pad,
                                                  filters,
                                                  Strides{1, 1},
                                                  CoordinateDiff{0, 0},
                                                  CoordinateDiff{1, 1},
                                                  Shape{1, 1});
        function_ref = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data, filters});
    }
}

PAD_TEST_BODY(NegativePadFusionNonZeroPadValue) {
    Shape data_shape{1, 3, 14, 14};
    {
        auto data = std::make_shared<Parameter>(element::i32, data_shape);
        auto pads_begin = Constant::create(element::i32, Shape{4}, {0, 0, 1, 1});
        auto pads_end = Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        auto pad_value = Constant::create(element::i32, Shape{}, {2});
        auto pad = pad_factory->create(data, pads_begin, pads_end, pad_value, op::PadMode::CONSTANT);
        auto filters = std::make_shared<Parameter>(element::i32, Shape{1, 3, 4, 4});
        auto conv = std::make_shared<Convolution>(pad,
                                                  filters,
                                                  Strides{1, 1},
                                                  CoordinateDiff{0, 0},
                                                  CoordinateDiff{1, 1},
                                                  Shape{1, 1});
        function = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data, filters});
        manager.register_pass<ov::pass::PadFusion>();
    }
    {
        auto data = std::make_shared<Parameter>(element::i32, data_shape);
        auto pads_begin = Constant::create(element::i32, Shape{4}, {0, 0, 1, 1});
        auto pads_end = Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        auto pad_value = Constant::create(element::i32, Shape{}, {2});
        auto pad = pad_factory->create(data, pads_begin, pads_end, pad_value, op::PadMode::CONSTANT);
        auto filters = std::make_shared<Parameter>(element::i32, Shape{1, 3, 4, 4});
        auto conv = std::make_shared<Convolution>(pad,
                                                  filters,
                                                  Strides{1, 1},
                                                  CoordinateDiff{0, 0},
                                                  CoordinateDiff{1, 1},
                                                  Shape{1, 1});
        function_ref = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data, filters});
    }
}

PAD_TEST_BODY(NegativePadFusionPadForBatchSize) {
    Shape data_shape{1, 3, 14, 14};
    {
        auto data = std::make_shared<Parameter>(element::i32, data_shape);
        auto pads_begin = Constant::create(element::i32, Shape{4}, {1, 0, 1, 1});
        auto pads_end = Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        auto pad_value = Constant::create(element::i32, Shape{}, {0});
        auto pad = pad_factory->create(data, pads_begin, pads_end, pad_value, op::PadMode::CONSTANT);
        auto filters = std::make_shared<Parameter>(element::i32, Shape{1, 3, 4, 4});
        auto conv = std::make_shared<Convolution>(pad,
                                                  filters,
                                                  Strides{1, 1},
                                                  CoordinateDiff{0, 0},
                                                  CoordinateDiff{1, 1},
                                                  Shape{1, 1});
        function = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data, filters});
        manager.register_pass<ov::pass::PadFusion>();
    }
    {
        auto data = std::make_shared<Parameter>(element::i32, data_shape);
        auto pads_begin = Constant::create(element::i32, Shape{4}, {1, 0, 1, 1});
        auto pads_end = Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        auto pad_value = Constant::create(element::i32, Shape{}, {0});
        auto pad = pad_factory->create(data, pads_begin, pads_end, pad_value, op::PadMode::CONSTANT);
        auto filters = std::make_shared<Parameter>(element::i32, Shape{1, 3, 4, 4});
        auto conv = std::make_shared<Convolution>(pad,
                                                  filters,
                                                  Strides{1, 1},
                                                  CoordinateDiff{0, 0},
                                                  CoordinateDiff{1, 1},
                                                  Shape{1, 1});
        function_ref = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data, filters});
    }
}

PAD_TEST_BODY(NegativePadFusionAvgPoolExcludePadNonZeroPads) {
    Shape data_shape{1, 3, 14, 14};
    {
        auto data = std::make_shared<Parameter>(element::i32, data_shape);
        auto pads_begin = Constant::create(element::i32, Shape{4}, {0, 0, 1, 1});
        auto pads_end = Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        auto pad = pad_factory->create(data, pads_begin, pads_end, op::PadMode::CONSTANT);
        auto avg_pool = std::make_shared<AvgPool>(pad,
                                                  Strides{1, 1},
                                                  Shape{0, 0},
                                                  Shape{1, 1},
                                                  Shape{4, 4},
                                                  true,
                                                  op::RoundingType::FLOOR);
        function = std::make_shared<Model>(NodeVector{avg_pool}, ParameterVector{data});
        manager.register_pass<ov::pass::PadFusion>();
    }
    {
        auto data = std::make_shared<Parameter>(element::i32, data_shape);
        auto pads_begin = Constant::create(element::i32, Shape{4}, {0, 0, 1, 1});
        auto pads_end = Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        auto pad = pad_factory->create(data, pads_begin, pads_end, op::PadMode::CONSTANT);
        auto avg_pool = std::make_shared<AvgPool>(pad,
                                                  Strides{1, 1},
                                                  Shape{0, 0},
                                                  Shape{1, 1},
                                                  Shape{4, 4},
                                                  true,
                                                  op::RoundingType::FLOOR);
        function_ref = std::make_shared<Model>(NodeVector{avg_pool}, ParameterVector{data});
    }
}

PAD_TEST_BODY(NegativePadFusionConvolutionBackpropDataTooSmallPad) {
    Shape data_shape{1, 3, 14, 14};
    {
        auto data = std::make_shared<Parameter>(element::f32, data_shape);

        auto pads_begin = Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        auto pads_end = Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        auto pad = pad_factory->create(data, pads_begin, pads_end, op::PadMode::CONSTANT);

        auto filters = std::make_shared<Parameter>(element::f32, Shape{3, 2, 5, 5});
        auto conv = std::make_shared<ConvolutionBackpropData>(pad,
                                                              filters,
                                                              Strides{1, 1},
                                                              CoordinateDiff{1, 1},
                                                              CoordinateDiff{1, 1},
                                                              Shape{1, 1});

        function = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data, filters});
        manager.register_pass<ov::pass::PadFusion>();
    }
    {
        auto data = std::make_shared<Parameter>(element::f32, data_shape);

        auto pads_begin = Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        auto pads_end = Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        auto pad = pad_factory->create(data, pads_begin, pads_end, op::PadMode::CONSTANT);

        auto filters = std::make_shared<Parameter>(element::f32, Shape{3, 2, 5, 5});
        auto conv = std::make_shared<ConvolutionBackpropData>(pad,
                                                              filters,
                                                              Strides{1, 1},
                                                              CoordinateDiff{1, 1},
                                                              CoordinateDiff{1, 1},
                                                              Shape{1, 1});

        function_ref = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data, filters});
    }
}

PAD_TEST_BODY(NegativePadPreservation) {
    Shape data_shape{1, 3, 14, 14};
    {
        auto data = std::make_shared<Parameter>(element::i32, data_shape);
        auto pads_begin = Constant::create(element::i32, Shape{4}, {0, 0, -1, -1});
        auto pads_end = Constant::create(element::i32, Shape{4}, {0, 0, -1, -1});
        auto pad = pad_factory->create(data, pads_begin, pads_end, op::PadMode::CONSTANT);
        auto filters = std::make_shared<Parameter>(element::i32, Shape{1, 3, 4, 4});
        auto conv = std::make_shared<Convolution>(pad,
                                                  filters,
                                                  Strides{1, 1},
                                                  CoordinateDiff{0, 0},
                                                  CoordinateDiff{1, 1},
                                                  Shape{1, 1});
        function = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data, filters});
        manager.register_pass<ov::pass::PadFusion>();
    }
    // Reference function is equal to function
}

namespace {

#undef CREATE_MODEL_FACTORY
#define CREATE_MODEL_FACTORY(type_name) std::make_shared<type_name>()

std::vector<TestModelFactoryPtr> model_factories = {
    CREATE_MODEL_FACTORY(PadElimination),
    CREATE_MODEL_FACTORY(PadFusionAvgPoolExcludePad),
    CREATE_MODEL_FACTORY(PadFusionAvgPoolDontExcludePad),
    CREATE_MODEL_FACTORY(PadFusionConvolution),
    CREATE_MODEL_FACTORY(PadFusionConvolutionBackpropData),
    CREATE_MODEL_FACTORY(PadFusionGroupConvolution),
    CREATE_MODEL_FACTORY(PadFusionGroupConvolutionBackpropData),
    CREATE_MODEL_FACTORY(PadFusionAvgPoolNonConstPadValue),
    CREATE_MODEL_FACTORY(PadFusionConvolutionNonConstPadValue),
    CREATE_MODEL_FACTORY(PadFusionConvolutionBackpropDataNonConstPadValue),
    CREATE_MODEL_FACTORY(PadFusionGroupConvolutionNonConstPadValue),
    CREATE_MODEL_FACTORY(PadFusionGroupConvolutionBackpropDataNonConstPadValue),
    CREATE_MODEL_FACTORY(NegativePadFusionNonConstantPadMode),
    CREATE_MODEL_FACTORY(NegativePadFusionNonZeroPadValue),
    CREATE_MODEL_FACTORY(NegativePadFusionPadForBatchSize),
    CREATE_MODEL_FACTORY(NegativePadFusionAvgPoolExcludePadNonZeroPads),
    CREATE_MODEL_FACTORY(NegativePadFusionConvolutionBackpropDataTooSmallPad),
    CREATE_MODEL_FACTORY(NegativePadPreservation),
    CREATE_MODEL_FACTORY(NegativePadElimination),
    CREATE_MODEL_FACTORY(NegativePadFusionAvgPoolExcludePad),
    CREATE_MODEL_FACTORY(NegativePadFusionAvgPoolDontExcludePad),
    CREATE_MODEL_FACTORY(NegativePadFusionConvolution),
    CREATE_MODEL_FACTORY(NegativePadFusionGroupConvolution)};

#undef CREATE_PAD_FACTORY
#define CREATE_PAD_FACTORY(type_name, type_str) CreatePadFactory<type_name>(type_str)

std::vector<PadFactoryPtr> pad_factories = {CREATE_PAD_FACTORY(ov::op::v1::Pad, "op_v1_Pad"),
                                            CREATE_PAD_FACTORY(ov::op::v12::Pad, "op_v12_Pad")};

}  // namespace

INSTANTIATE_TEST_SUITE_P(PadTestSuite,
                         PadTestFixture,
                         ::testing::Combine(::testing::ValuesIn(pad_factories), ::testing::ValuesIn(model_factories)),
                         PadTestFixture::get_test_name);
