// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/common_optimizations/ric_fusion.hpp>
#include <transformations/init_node_info.hpp>
#include <ngraph_functions/utils/ngraph_helpers.hpp>
#include <openvino/core/preprocess/pre_post_process.hpp>
#include <transformations/serialize.hpp>
#include <ngraph/pass/constant_folding.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ngraph;

TEST_F(TransformationTestsF, RICFusionSimple) {
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 3, 64, 64 });
        auto relu = std::make_shared<ngraph::opset8::Relu>(input);
        auto conv = std::make_shared<ngraph::opset8::Convolution>(relu, opset8::Constant::create(element::f32, Shape{6, 3, 3, 3}, {0.1}),
                                                                  ov::Strides{1, 1}, ov::CoordinateDiff{0, 0}, ov::CoordinateDiff{0, 0}, ov::Strides{1, 1});

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ conv }, ngraph::ParameterVector{ input });

        using namespace ov::preprocess;
        PrePostProcessor p(function);
        p.input().tensor().set_layout("NCHW");
        p.input(0).preprocess().reverse_channels();
        p.build();

        manager.register_pass<ngraph::pass::Serialize>("/tmp/before.xml", "/tmp/before.bin");
        manager.register_pass<ngraph::pass::ConstantFolding>();
        manager.register_pass<ngraph::pass::ReverseInputChannelsFusion>();
        manager.register_pass<ngraph::pass::Serialize>("/tmp/after.xml", "/tmp/after.bin");
        manager.register_pass<ngraph::pass::ConstantFolding>();
    }

    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 3, 64, 64 });
        auto relu = std::make_shared<ngraph::opset8::Relu>(input);
        auto conv = std::make_shared<ngraph::opset8::Convolution>(relu, opset8::Constant::create(element::f32, Shape{6, 3, 3, 3}, {0.1}),
                                                                  ov::Strides{1, 1}, ov::CoordinateDiff{0, 0}, ov::CoordinateDiff{0, 0}, ov::Strides{1, 1});

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ conv }, ngraph::ParameterVector{ input });
    }
}

TEST_F(TransformationTestsF, RICFusionHard) {
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 3, 64, 64 });
        auto relu = std::make_shared<ngraph::opset8::Relu>(input);

        auto input2 = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 3, 64, 64 });
        auto eltwise = std::make_shared<ngraph::opset8::Add>(relu, input2);

        auto conv = std::make_shared<ngraph::opset8::Convolution>(eltwise, opset8::Constant::create(element::f32, Shape{6, 3, 3, 3}, {0.1}),
                                                                  ov::Strides{1, 1}, ov::CoordinateDiff{0, 0}, ov::CoordinateDiff{0, 0}, ov::Strides{1, 1});

        auto conv2 = std::make_shared<ngraph::opset8::Convolution>(input2, opset8::Constant::create(element::f32, Shape{6, 3, 3, 3}, {0.1}),
                                                                  ov::Strides{1, 1}, ov::CoordinateDiff{0, 0}, ov::CoordinateDiff{0, 0}, ov::Strides{1, 1});

        auto concat = std::make_shared<opset8::Concat>(OutputVector{conv, conv2}, 1);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ concat }, ngraph::ParameterVector{ input, input2 });

        using namespace ov::preprocess;
        PrePostProcessor p(function);
        p.input(0).tensor().set_layout("NCHW");
        p.input(1).tensor().set_layout("NCHW");
        p.input(0).preprocess().reverse_channels();
        p.input(1).preprocess().reverse_channels();
        p.build();

        manager.register_pass<ngraph::pass::Serialize>("/tmp/before.xml", "/tmp/before.bin");
        manager.register_pass<ngraph::pass::ConstantFolding>();
        manager.register_pass<ngraph::pass::ReverseInputChannelsFusion>();
        manager.register_pass<ngraph::pass::Serialize>("/tmp/after.xml", "/tmp/after.bin");
        manager.register_pass<ngraph::pass::ConstantFolding>();
    }
}