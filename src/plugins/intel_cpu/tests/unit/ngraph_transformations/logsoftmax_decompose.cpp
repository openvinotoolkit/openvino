// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <openvino/opsets/opset1.hpp>
#include <openvino/opsets/opset8.hpp>
#include <ngraph_transformations/op/fully_connected.hpp>
#include <ngraph_transformations/convert_logsoftmax.hpp>
#include <ngraph_transformations/fc_bias_fusion.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <openvino/pass/manager.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ov::intel_cpu;

TEST_F(TransformationTestsF, ConvertLogSoftmax1) {
    int compute_axis = 1;
    {
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{ 3, 2, 2 });
        auto logsoftmax = std::make_shared<ov::opset8::LogSoftmax>(input, compute_axis);
        function = std::make_shared<ov::Model>(ov::NodeVector{ logsoftmax }, ov::ParameterVector{ input });
        manager.register_pass<ConvertLogSoftmax>();
    }

    {
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{ 3, 2, 2 });
        auto axis = std::make_shared<ov::opset8::Constant>(ov::element::i64, ov::Shape{}, compute_axis);
        auto xMax = std::make_shared<ov::opset8::ReduceMax>(input, axis, true);
        auto subtract = std::make_shared<ov::opset8::Subtract>(input, xMax);
        auto exp = std::make_shared<ov::opset8::Exp>(subtract);
        auto s = std::make_shared<ov::opset8::ReduceSum>(exp, axis, true);
        auto log = std::make_shared<ov::opset8::Log>(s);
        auto result = std::make_shared<ov::opset8::Subtract>(subtract, log);
        function_ref = std::make_shared<ov::Model>(ov::NodeVector{ result }, ov::ParameterVector{ input });
    }
}

TEST_F(TransformationTestsF, ConvertLogSoftmaxSplit) {
    int compute_axis = 1;
    auto make_split = [](const std::shared_ptr<ov::Node>& input) {
        auto split_axis = std::make_shared<ov::opset8::Constant>(ov::element::i64, ov::Shape{}, 1);
        auto split_length = std::make_shared<ov::opset8::Constant>(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{4, 6});
        auto split = std::make_shared<ov::opset8::VariadicSplit>(input, split_axis, split_length);
        return split;
    };

    {
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{ 2, 10 });
        auto split = make_split(input);
        auto logsoftmax = std::make_shared<ov::opset8::LogSoftmax>(split->output(1), compute_axis);
        auto scale = std::make_shared<ov::opset8::Constant>(ov::element::f32, ov::Shape{}, 1);
        auto multiply = std::make_shared<ov::opset8::Multiply>(split->output(0), scale);

        function = std::make_shared<ov::Model>(ov::NodeVector{multiply, logsoftmax}, ov::ParameterVector{ input });
        manager.register_pass<ConvertLogSoftmax>();
    }

    {
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{ 2, 10 });
        auto split = make_split(input);
        auto axis = std::make_shared<ov::opset8::Constant>(ov::element::i64, ov::Shape{}, compute_axis);
        auto xMax = std::make_shared<ov::opset8::ReduceMax>(split->output(1), axis, true);
        auto subtract = std::make_shared<ov::opset8::Subtract>(split->output(1), xMax);
        auto exp = std::make_shared<ov::opset8::Exp>(subtract);
        auto s = std::make_shared<ov::opset8::ReduceSum>(exp, axis, true);
        auto log = std::make_shared<ov::opset8::Log>(s);
        auto result = std::make_shared<ov::opset8::Subtract>(subtract, log);
        auto scale = std::make_shared<ov::opset8::Constant>(ov::element::f32, ov::Shape{}, 1);
        auto multiply = std::make_shared<ov::opset8::Multiply>(split->output(0), scale);
        function_ref = std::make_shared<ov::Model>(ov::NodeVector{ multiply, result }, ov::ParameterVector{ input });
    }
}