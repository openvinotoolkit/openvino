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

TEST(TransformationTests, ConvertLogSoftmax1) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    int compute_axis = 1;
    {
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{ 3, 2, 2 });
        auto logsoftmax = std::make_shared<ov::opset8::LogSoftmax>(input, compute_axis);

        f = std::make_shared<ov::Model>(ov::NodeVector{ logsoftmax }, ov::ParameterVector{ input });
        ov::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<ConvertLogSoftmax>();
        m.run_passes(f);
    }

    {
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{ 3, 2, 2 });
        auto axis = std::make_shared<ov::opset8::Constant>(ov::element::i64, ov::Shape{}, compute_axis);
        auto xMax = std::make_shared<ov::opset8::ReduceMax>(input->get_default_output(), axis->get_default_output(), true);
        auto subtract = std::make_shared<ov::opset8::Subtract>(input->get_default_output(), xMax->get_default_output());
        auto tmp = std::make_shared<ov::opset8::Exp>(subtract->get_default_output());
        auto s = std::make_shared<ov::opset8::ReduceSum>(tmp->get_default_output(), axis->get_default_output(), true);
        auto log = std::make_shared<ov::opset8::Log>(s);
        auto result = std::make_shared<ov::opset8::Subtract>(subtract, log);

        f_ref = std::make_shared<ov::Model>(ov::NodeVector{ result }, ov::ParameterVector{ input });
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}